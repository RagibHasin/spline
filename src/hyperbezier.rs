//! Representation and computation of hyperbeziers.

use kurbo::{
    common::GAUSS_LEGENDRE_COEFFS_32, Affine, CurveFitSample, ParamCurve, ParamCurveFit,
    ParamCurveNearest, Point, Vec2,
};

#[derive(Clone, Copy, Debug)]
pub struct HyperbezParams {
    a: f64,
    b: f64,
    c: f64,
    d: f64,

    th_a: f64,
    th_b: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Hyperbezier {
    params: HyperbezParams,
    p0: Point,
    p1: Point,
    scale_rot: Vec2,
}

impl HyperbezParams {
    /// Create a new hyperbezier with the given parameters.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        // TODO: numerical stability issues when c is very small
        // (which can happen for pure Euler spiral) or 4c-d^2 is
        // very small (hyperbola-like with very high curvature).
        let denom = 2. / (4. * c - d * d);
        let dd = d * denom;
        let th_a = (2. * b * c - d * a) * denom;
        let th_b = b * dd - a * (1. + 0.5 * d * dd) / c;
        HyperbezParams {
            a,
            b,
            c,
            d,
            th_a,
            th_b,
        }
    }

    /// Determine the angle for the given parameter.
    ///
    /// This can be interpreted as a Whewell representation of the
    /// curve. The `t` parameter ranges from 0 to 1, and the returned
    /// value is 0 for `t = 0`.
    pub fn theta(&self, t: f64) -> f64 {
        let q = self.c * t * t + self.d * t + 1.;
        (self.th_a * t + self.th_b) / q.sqrt() - self.th_b
    }

    pub fn kappa(&self, t: f64) -> f64 {
        let q = self.c * t * t + self.d * t + 1.;
        let q_sqrt = q.sqrt();
        (self.a * t + self.b) / q / q_sqrt
    }

    /// Evaluate the position of the raw curve.
    ///
    /// This is simply the integral of the Whewell representation,
    /// so that the total arc length is unit, and the initial tangent
    /// is horizontal.
    pub fn integrate(&self, t: f64) -> Vec2 {
        // TODO: improve accuracy by subdividing in near-cusp cases
        let mut xy = Vec2::ZERO;
        let u0 = 0.5 * t;
        for (wi, xi) in GAUSS_LEGENDRE_COEFFS_32 {
            let u = u0 + u0 * xi;
            xy += *wi * Vec2::from_angle(self.theta(u));
        }
        u0 * xy
    }

    pub fn a(&self) -> f64 {
        self.a
    }
    pub fn b(&self) -> f64 {
        self.b
    }
    pub fn c(&self) -> f64 {
        self.c
    }
    pub fn d(&self) -> f64 {
        self.d
    }
    pub fn th_a(&self) -> f64 {
        self.th_a
    }
    pub fn th_b(&self) -> f64 {
        self.th_b
    }
}

impl Hyperbezier {
    /// Create a new hyperbezier curve with given parameters and end points.
    pub fn from_points_params(params: HyperbezParams, p0: Point, p1: Point) -> Self {
        let uv = params.integrate(1.0);
        let uv_scaled = uv / uv.length_squared();
        let d = p1 - p0;
        let scale_rot = Vec2::new(uv_scaled.dot(d), uv_scaled.cross(d));
        Hyperbezier {
            params,
            p0,
            p1,
            scale_rot,
        }
    }

    pub fn params(&self) -> &HyperbezParams {
        &self.params
    }

    pub fn scale_rot(&self) -> Vec2 {
        self.scale_rot
    }

    pub fn theta(&self, t: f64) -> f64 {
        self.params.theta(t) + self.scale_rot.angle()
    }

    pub fn kappa(&self, t: f64) -> f64 {
        self.params.kappa(t) / self.scale_rot.length()
    }
}

impl ParamCurve for Hyperbezier {
    fn eval(&self, t: f64) -> Point {
        if t == 1.0 {
            self.p1
        } else {
            let s = self.scale_rot;
            let uv = self.params.integrate(t);
            self.p0 + Vec2::new(s.x * uv.x - s.y * uv.y, s.x * uv.y + s.y * uv.x)
        }
    }

    fn start(&self) -> Point {
        self.p0
    }

    fn end(&self) -> Point {
        self.p1
    }

    fn subsegment(&self, range: std::ops::Range<f64>) -> Self {
        let (t0, t1) = (range.start, range.end);
        let dt = t1 - t0;
        let a = self.params.a * dt;
        let b = self.params.b + self.params.a * t0;
        let c = self.params.c * dt * dt;
        let d = (self.params.d + 2. * self.params.c * t0) * dt;
        let e = self.params.c * t0 * t0 + self.params.d * t0 + 1.;
        let s = 1. / e;
        let ps = dt * s * s.sqrt();
        let params = HyperbezParams::new(a * ps, b * ps, c * s, d * s);
        let p0 = self.eval(t0);
        let p1 = self.eval(t1);
        Hyperbezier::from_points_params(params, p0, p1)
    }
}

impl ParamCurveFit for Hyperbezier {
    fn sample_pt_tangent(&self, t: f64, _sign: f64) -> CurveFitSample {
        let (p, tangent) = self.sample_pt_deriv(t);
        CurveFitSample { p, tangent }
    }

    fn sample_pt_deriv(&self, t: f64) -> (Point, Vec2) {
        let p = self.eval(t);
        let uv = Vec2::from_angle(self.params.theta(t));
        let s = self.scale_rot;
        let d = Vec2::new(s.x * uv.x - s.y * uv.y, s.x * uv.y + s.y * uv.x);
        (p, d)
    }

    fn break_cusp(&self, _: std::ops::Range<f64>) -> Option<f64> {
        None
    }
}

impl ParamCurveNearest for Hyperbezier {
    fn nearest(&self, p: Point, accuracy: f64) -> kurbo::Nearest {
        let p_local = Affine::translate(self.p0.to_vec2())
            .then_rotate(-self.scale_rot.angle())
            .then_scale(1. / self.scale_rot.length())
            * p;

        // 1. if theta1 < 2pi, check if p_local is in the sweep region between normal0 and normal1
        //   1a. if true, then subdivide and repeat from 1 for each half
        //   1b. if false, then either s = 0 or s = 1 is nearest, check and tell
        // 2. otherwise sibdivide for theta1 = 2pi and repeat from 1

        if self.params.theta(1.) >= std::f64::consts::TAU {}

        todo!()
    }
}
