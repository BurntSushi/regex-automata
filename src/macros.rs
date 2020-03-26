// A simple macro for defining bitfield accessors/mutators.
#[macro_export]
macro_rules! define_bool {
    ($bit:expr, $is_fn_name:ident, $set_fn_name:ident) => {
        fn $is_fn_name(&self) -> bool {
            self.bools & (0b1 << $bit) > 0
        }

        fn $set_fn_name(&mut self, yes: bool) {
            if yes {
                self.bools |= 1 << $bit;
            } else {
                self.bools &= !(1 << $bit);
            }
        }
    }
}
