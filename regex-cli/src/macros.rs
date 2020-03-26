/// A macro for running code over each of the possible StateID representations.
///
/// This macro effectively converts a runtime value specified by the user (via
/// the --state-id flag) to a type: u8, u16, u32 or u64. It then instantiates
/// the `run` function with that type and calls it with the given arguments.
#[cfg(feature = "state-sizes")]
#[macro_export]
macro_rules! each_state_size {
    ($state_size:expr, $run:ident, $($args:expr),*) => {{
        match $state_size {
            1 => $run::<u8>($($args),*),
            2 => $run::<u16>($($args),*),
            4 => $run::<u32>($($args),*),
            8 => $run::<u64>($($args),*),
            _ => unreachable!(),
        }
    }};
}

/// A macro for running code over each of the possible StateID representations.
///
/// When the state-sizes feature is disabled, then only a pointer-sized StateID
/// representation is supported. Any other will return an error.
///
/// This is useful for decreasing compiling times, since instantiating the same
/// function for every state ID representation can lead to a lot of code bloat.
#[cfg(not(feature = "state-sizes"))]
#[macro_export]
macro_rules! each_state_size {
    ($state_size:expr, $run:ident, $($args:expr),*) => {{
        if $state_size != std::mem::size_of::<usize>() {
            anyhow::bail!(
                "only {} is supported as a state id size \
                 (to lift this restriction, compile with the state-sizes \
                 feature)",
                std::mem::size_of::<usize>(),
            );
        }
        $run::<usize>($($args),*)
    }};
}
