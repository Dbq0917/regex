use alloc::{vec, vec::Vec};

use regex_syntax::hir::Hir;

use crate::{meta::regex::RegexInfo, util::search::MatchKind};

#[cfg(feature = "perf-literal-multisubstring")]
use crate::util::{primitives::PatternID, search::PatternSet};

/// Pull out an alternation of literals from the given sequence of HIR
/// expressions.
///
/// There are numerous ways for this to fail. Generally, this only applies
/// to regexes of the form 'foo|bar|baz|...|quux'. It can also fail if there
/// are "too few" alternates, in which case, the regex engine is likely faster.
///
/// And currently, this only returns something when 'hirs.len() == 1'.
pub(crate) fn alternation_literals(
    info: &RegexInfo,
    hirs: &[&Hir],
) -> Option<Vec<Vec<u8>>> {
    use regex_syntax::hir::{HirKind, Literal};

    // Might as well skip the work below if we know we can't build an
    // Aho-Corasick searcher.
    if !cfg!(feature = "perf-literal-multisubstring") {
        return None;
    }
    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    if hirs.len() != 1
        || !info.props()[0].look_set().is_empty()
        || info.props()[0].explicit_captures_len() > 0
        || !info.props()[0].is_alternation_literal()
        || info.config().get_match_kind() != MatchKind::LeftmostFirst
    {
        return None;
    }
    let hir = &hirs[0];
    let alts = match *hir.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return None, // one literal isn't worth it
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(Literal(ref bytes)) => {
                lit.extend_from_slice(bytes)
            }
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(Literal(ref bytes)) => {
                            lit.extend_from_slice(bytes);
                        }
                        _ => unreachable!("expected literal, got {e:?}"),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {alt:?}"),
        }
        lits.push(lit);
    }
    // Why do this? Well, when the number of literals is small, it's likely
    // that we'll use the lazy DFA which is in turn likely to be faster than
    // Aho-Corasick in such cases. Primarily because Aho-Corasick doesn't have
    // a "lazy DFA" but either a contiguous NFA or a full DFA. We rarely use
    // the latter because it is so hungry (in time and space), and the former
    // is decently fast, but not as fast as a well oiled lazy DFA.
    //
    // However, once the number starts getting large, the lazy DFA is likely
    // to start thrashing because of the modest default cache size. When
    // exactly does this happen? Dunno. But at whatever point that is (we make
    // a guess below based on ad hoc benchmarking), we'll want to cut over to
    // Aho-Corasick, where even the contiguous NFA is likely to do much better.
    if lits.len() < 3000 {
        debug!("skipping Aho-Corasick because there are too few literals");
        return None;
    }
    Some(lits)
}

/// A large set of exact literals with a mapping back to regex pattern IDs.
///
/// This is only used to accelerate reporting which patterns match. It is not
/// used as a prefilter, since a prefilter can only report a span and loses
/// the correspondence between a literal and its pattern ID.
#[cfg(feature = "perf-literal-multisubstring")]
#[derive(Debug)]
pub(crate) struct LiteralSet {
    ac: aho_corasick::AhoCorasick,
    pids: Vec<PatternID>,
}

#[cfg(feature = "perf-literal-multisubstring")]
impl LiteralSet {
    /// Build a literal set from the given HIRs if every pattern is a non-empty
    /// exact literal.
    ///
    /// This is deliberately conservative. In particular, patterns with look
    /// around assertions (including anchors), character classes, repetitions
    /// that aren't fully unfolded into literals, or empty literals are
    /// rejected. Those patterns require the full regex engine to preserve
    /// their semantics.
    pub(crate) fn new(info: &RegexInfo, hirs: &[&Hir]) -> Option<LiteralSet> {
        use regex_syntax::hir::literal::{ExtractKind, Extractor};

        // Small sets are already fast enough with the normal engines, and
        // building another automaton isn't worth it.
        const MIN_PATTERNS: usize = 100;

        if info.config().get_match_kind() != MatchKind::All
            || hirs.len() < MIN_PATTERNS
        {
            return None;
        }

        let mut needles = Vec::new();
        let mut pids = Vec::new();
        for (pid, hir) in hirs.iter().enumerate() {
            if !info.props()[pid].look_set().is_empty() {
                return None;
            }
            let mut extractor = Extractor::new();
            extractor.kind(ExtractKind::Prefix);
            let prefixes = extractor.extract(hir);
            if !prefixes.is_exact() {
                return None;
            }
            let literals = prefixes.literals()?;
            if literals.iter().any(|lit| lit.is_empty()) {
                return None;
            }
            let pid = PatternID::new(pid).ok()?;
            for lit in literals {
                needles.push(lit.as_bytes().to_vec());
                pids.push(pid);
            }
        }

        let ac_kind = if needles.len() <= 500 {
            aho_corasick::AhoCorasickKind::DFA
        } else {
            aho_corasick::AhoCorasickKind::ContiguousNFA
        };
        let ac = aho_corasick::AhoCorasick::builder()
            .kind(Some(ac_kind))
            .match_kind(aho_corasick::MatchKind::Standard)
            .start_kind(aho_corasick::StartKind::Both)
            .prefilter(false)
            .build(&needles)
            .ok()?;
        Some(LiteralSet { ac, pids })
    }

    /// Returns true when the given search can safely use this literal set.
    ///
    /// Anchored and earliest searches have semantics that differ from an
    /// ordinary unanchored literal search, so they use the regex engine.
    #[inline]
    pub(crate) fn is_search_eligible(input: &crate::Input<'_>) -> bool {
        !input.is_done()
            && matches!(input.get_anchored(), crate::Anchored::No)
            && !input.get_earliest()
    }

    #[inline]
    pub(crate) fn is_match(&self, input: &crate::Input<'_>) -> bool {
        let input = aho_corasick::Input::new(input.haystack())
            .span(input.start()..input.end());
        self.ac.is_match(input)
    }

    pub(crate) fn which_overlapping_matches(
        &self,
        input: &crate::Input<'_>,
        patset: &mut PatternSet,
    ) {
        let input = aho_corasick::Input::new(input.haystack())
            .span(input.start()..input.end());
        for m in self.ac.find_overlapping_iter(input) {
            let _ = patset.try_insert(self.pids[m.pattern().as_usize()]);
            if patset.is_full() {
                break;
            }
        }
    }

    pub(crate) fn memory_usage(&self) -> usize {
        self.ac.memory_usage()
            + self.pids.len() * core::mem::size_of::<PatternID>()
    }
}

#[cfg(all(test, feature = "perf-literal-multisubstring"))]
mod tests {
    use alloc::{format, string::String, vec::Vec};

    use crate::{
        meta::regex::Config, util::syntax, Anchored, Input, MatchKind,
    };

    use super::*;

    fn literal_set(patterns: &[String]) -> Option<LiteralSet> {
        let hirs = syntax::parse_many(patterns).unwrap();
        let hirs: Vec<&Hir> = hirs.iter().collect();
        let info = crate::meta::regex::RegexInfo::new(
            Config::new().match_kind(MatchKind::All),
            &hirs,
        );
        LiteralSet::new(&info, &hirs)
    }

    fn patterns(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| format!("service-common-prefix-rule-{i:04}"))
            .collect()
    }

    #[test]
    fn large_literal_set_reports_pattern_ids() {
        let patterns = patterns(2_000);
        let set = literal_set(&patterns).unwrap();
        let hay = "x service-common-prefix-rule-0042 \
                   service-common-prefix-rule-1999 y";
        let input = Input::new(hay);
        let mut patset = PatternSet::new(patterns.len());
        set.which_overlapping_matches(&input, &mut patset);
        let got: Vec<usize> =
            patset.iter().map(|pid| pid.as_usize()).collect();
        assert_eq!(vec![42, 1999], got);
        assert!(set.is_match(&input));
        assert!(!set.is_match(&Input::new("no rules match here")));
    }

    #[test]
    fn large_literal_set_respects_search_span() {
        let patterns = patterns(2_000);
        let set = literal_set(&patterns).unwrap();
        let hay = "service-common-prefix-rule-0001 x";
        let input = Input::new(hay).span(1..hay.len());
        assert!(!set.is_match(&input));
        assert!(!LiteralSet::is_search_eligible(
            &input.anchored(Anchored::Yes)
        ));
    }

    #[test]
    fn large_set_with_nonliteral_pattern_is_rejected() {
        let mut patterns = patterns(100);
        patterns.push(String::from(r"service-common-prefix-rule-\d+"));
        assert!(literal_set(&patterns).is_none());
    }

    #[test]
    fn large_set_with_anchor_is_rejected() {
        let mut patterns = patterns(100);
        patterns.push(String::from(r"^service-common-prefix-rule"));
        assert!(literal_set(&patterns).is_none());
    }
}
