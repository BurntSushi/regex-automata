#[macro_export]
macro_rules! is_match {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start();
        if $dfa.is_possible_match_state(state) {
            return $dfa.is_match_state(state);
        }
        for &b in $bytes.iter() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                return $dfa.is_match_state(state);
            }
        }
        false
    }}
}

#[macro_export]
macro_rules! shortest_match {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start();
        if $dfa.is_possible_match_state(state) {
            return if $dfa.is_dead(state) { None } else { Some(0) };
        }
        for (i, &b) in $bytes.iter().enumerate() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                return if $dfa.is_dead(state) { None } else { Some(i + 1) };
            }
        }
        None
    }}
}

#[macro_export]
macro_rules! find {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start;
        let mut last_match =
            if $dfa.is_dead(state) {
                return None;
            } else if $dfa.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in $bytes.iter().enumerate() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                if $dfa.is_dead(state) {
                    return last_match;
                }
                last_match = Some(i + 1);
            }
        }
        last_match
    }}
}

#[macro_export]
macro_rules! rfind {
    ($dfa:ident, $bytes:ident, $trans:ident) => {{
        let mut state = $dfa.start;
        let mut last_match =
            if $dfa.is_dead(state) {
                return None;
            } else if $dfa.is_match_state(state) {
                Some(0)
            } else {
                None
            };
        for (i, &b) in $bytes.iter().enumerate().rev() {
            state = unsafe { $dfa.$trans(state, b) };
            if $dfa.is_possible_match_state(state) {
                if $dfa.is_dead(state) {
                    return last_match;
                }
                last_match = Some(i);
            }
        }
        last_match
    }}
}
