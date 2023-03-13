use std::fmt::Debug;

use lexopt::{Arg, Parser, ValueExt};

use crate::args::{self, Usage};

pub mod input;
pub mod patterns;
pub mod syntax;
pub mod thompson;

pub trait Configurable: Debug + 'static {
    fn configure(
        &mut self,
        p: &mut Parser,
        arg: &mut Arg,
    ) -> anyhow::Result<bool>;

    fn usage(&self) -> &[Usage];
}

pub fn configure(
    p: &mut Parser,
    usage: &str,
    targets: &mut [&mut dyn Configurable],
) -> anyhow::Result<()> {
    while let Some(mut arg) = p.next()? {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                let mut usages = vec![];
                for t in targets.iter() {
                    usages.extend_from_slice(t.usage());
                }
                usages.sort_by_key(|u| u.format);
                let options = if arg == Arg::Short('h') {
                    Usage::short(&usages)
                } else {
                    Usage::long(&usages)
                };
                let usage = usage.replace("%options%", &options);
                anyhow::bail!("{}", usage);
            }
            _ => {}
        }
        // We do this little dance to disentangle the lifetime of 'p' from the
        // lifetime on 'arg'. The cost is that we have to clone all long flag
        // names to give it a place to live that isn't tied to 'p'. Annoying,
        // but not the end of the world.
        let long_flag: Option<String> = match arg {
            Arg::Long(name) => Some(name.to_string()),
            _ => None,
        };
        let mut arg = match long_flag {
            Some(ref flag) => Arg::Long(flag),
            None => match arg {
                Arg::Short(c) => Arg::Short(c),
                Arg::Long(_) => unreachable!(),
                Arg::Value(value) => Arg::Value(value),
            },
        };
        // OK, now ask all of our targets whether they want this argument.
        let mut recognized = false;
        for t in targets.iter_mut() {
            if t.configure(p, &mut arg)? {
                recognized = true;
                break;
            }
        }
        if !recognized {
            return Err(arg.unexpected().into());
        }
    }
    Ok(())
}
