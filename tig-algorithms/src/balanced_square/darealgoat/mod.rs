use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::{Duration, Instant};
use tig_challenges::balanced_square::*;

const PM: u64 = 350;
const MI: f64 = 1e-7;
const GV: f64 = 1.0;
const TB: u64 = 6000;
const ST: u64 = 700;
const RS: bool = true;

#[inline(always)] fn rci(r: usize, c: usize, n: usize) -> usize { r * n + c }
#[inline(always)] fn irc(i: usize, n: usize) -> (usize, usize) { (i / n, i % n) }
#[inline(always)] fn rl(r: usize) -> usize { r }
#[inline(always)] fn cl(n: usize, c: usize) -> usize { n + c }
#[inline(always)] fn md(n: usize) -> usize { 2 * n }
#[inline(always)] fn ad(n: usize) -> usize { 2 * n + 1 }
#[inline(always)] fn omd(r: usize, c: usize) -> bool { r == c }
#[inline(always)] fn oad(r: usize, c: usize, n: usize) -> bool { r + c == n - 1 }
#[inline(always)]
fn varf(s: i64, ss: i64, im: f64) -> f64 {
    let m = (s as f64) * im;
    (ss as f64) * im - m * m
}

fn seed(nums: &[i32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..nums.len()).collect();
    idx.sort_unstable_by_key(|&i| std::cmp::Reverse(nums[i]));
    let mut a = vec![0usize; n * n];
    let mut k = 0;
    for r in 0..n {
        if r % 2 == 0 {
            for c in 0..n { a[rci(r, c, n)] = idx[k]; k += 1; }
        } else {
            for c in (0..n).rev() { a[rci(r, c, n)] = idx[k]; k += 1; }
        }
    }
    a
}

fn init(nums: &[i32], n: usize, a: &[usize]) -> (Vec<i64>, i64, i64) {
    let m = 2 * n + 2;
    let mut s = vec![0i64; m];
    for r in 0..n {
        for c in 0..n {
            let v = nums[a[rci(r, c, n)]] as i64;
            s[rl(r)] += v;
            s[cl(n, c)] += v;
            if omd(r, c) { s[md(n)] += v; }
            if oad(r, c, n) { s[ad(n)] += v; }
        }
    }
    let sum: i64 = s.iter().sum();
    let sq: i64 = s.iter().map(|&x| x * x).sum();
    (s, sum, sq)
}

fn aff(n: usize, p: usize, q: usize, buf: &mut [usize; 8]) -> usize {
    let (pr, pc) = irc(p, n);
    let (qr, qc) = irc(q, n);
    let mut k = 0usize;
    let mut add = |x: usize| { for i in 0..k { if buf[i] == x { return; } } buf[k] = x; k += 1; };
    add(rl(pr));
    add(cl(n, pc));
    if omd(pr, pc) { add(md(n)); }
    if oad(pr, pc, n) { add(ad(n)); }
    add(rl(qr));
    add(cl(n, qc));
    if omd(qr, qc) { add(md(n)); }
    if oad(qr, qc, n) { add(ad(n)); }
    k
}

fn scr(
    nums: &[i32], n: usize, a: &[usize], p: usize, q: usize,
    ls: &[i64], s: i64, ss: i64, im: f64, tmp: &mut [usize; 8],
) -> (f64, i64, i64, usize, [i64; 8], [i64; 8]) {
    let vp = nums[a[p]] as i64;
    let vq = nums[a[q]] as i64;
    let dv = vq - vp;

    let k = aff(n, p, q, tmp);
    let (pr, pc) = irc(p, n);
    let (qr, qc) = irc(q, n);

    let mut ns = s;
    let mut nss = ss;
    let mut ov = [0i64; 8];
    let mut nv = [0i64; 8];

    for i in 0..k {
        let l = tmp[i];
        let pin = match l {
            x if x < n => x == pr,
            x if x < 2 * n => (x - n) == pc,
            x if x == 2 * n => pr == pc,
            _ => pr + pc == n - 1,
        };
        let qin = match l {
            x if x < n => x == qr,
            x if x < 2 * n => (x - n) == qc,
            x if x == 2 * n => qr == qc,
            _ => qr + qc == n - 1,
        };
        let d = if pin && !qin { dv } else if !pin && qin { -dv } else { 0 };
        let old = ls[l];
        let newl = old + d;
        ov[i] = old;
        nv[i] = newl;
        if d != 0 {
            ns += d;
            nss += newl * newl - old * old;
        }
    }
    (varf(ns, nss, im), ns, nss, k, ov, nv)
}

#[inline(always)]
fn apply(
    a: &mut [usize], p: usize, q: usize,
    ls: &mut [i64], tmp: &[usize; 8], k: usize,
    ov: &[i64; 8], nv: &[i64; 8],
) {
    a.swap(p, q);
    for i in 0..k {
        let l = tmp[i];
        if ov[i] != nv[i] { ls[l] = nv[i]; }
    }
}

#[inline(always)]
fn pair(rng: &mut SmallRng, l: usize) -> (usize, usize) {
    let a = rng.gen_range(0..l);
    let mut b = rng.gen_range(0..l);
    if a == b { b = (b + 1) % l; }
    (a, b)
}

fn to2d(a: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut o = vec![vec![0usize; n]; n];
    for r in 0..n { for c in 0..n { o[r][c] = a[rci(r, c, n)]; } }
    o
}

pub fn solve_challenge(
    ch: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(ch.seed);
    let n = (ch.numbers.len() as f64).sqrt() as usize;
    let m = 2 * n + 2;
    let im = 1.0 / (m as f64);

    let mut a = seed(&ch.numbers, n);
    let (mut ls, mut s, mut ss) = init(&ch.numbers, n, &a);
    let mut cv = varf(s, ss, im);

    save(&Solution { arrangement: to2d(&a, n) })?;
    if cv <= GV { return Ok(()); }

    let mut ba = a.clone();
    let mut bv = cv;
    let mut lsav = Instant::now();
    let st_d = Duration::from_millis(ST);
    let mut lbt = Instant::now();
    let ts = Instant::now();
    let tb = Duration::from_millis(TB);
    let mut t = 1200.0f64;
    let cool = 0.99975f64;
    let mut tmp: [usize; 8] = [0; 8];

    while ts.elapsed() < tb {
        let (p, q) = if RS { pair(&mut rng, n * n) } else { pair(&mut rng, n * n) };
        let (nv, ns, nss, k, ov, nlv) = scr(&ch.numbers, n, &a, p, q, &ls, s, ss, im, &mut tmp);
        let d = nv - cv;
        let ok = d <= 0.0 || rng.gen::<f64>().ln() < -d / t.max(1e-9);
        if ok {
            apply(&mut a, p, q, &mut ls, &tmp, k, &ov, &nlv);
            s = ns;
            ss = nss;
            cv = nv;
            if cv + MI < bv {
                bv = cv;
                ba.copy_from_slice(&a);
                lbt = Instant::now();
                if bv <= GV {
                    let _ = save(&Solution { arrangement: to2d(&ba, n) });
                    return Ok(());
                }
                if lsav.elapsed() >= st_d {
                    lsav = Instant::now();
                    let _ = save(&Solution { arrangement: to2d(&ba, n) });
                }
            }
        }
        t *= cool;
        if t < 0.05 { t = 15.0; }
        if lbt.elapsed() >= Duration::from_millis(PM) {
            let _ = save(&Solution { arrangement: to2d(&ba, n) });
            return Ok(());
        }
    }
    let _ = save(&Solution { arrangement: to2d(&ba, n) });
    Ok(())
}
