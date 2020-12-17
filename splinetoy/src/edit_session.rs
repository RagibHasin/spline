use std::sync::Arc;

use druid::kurbo::BezPath;
use druid::{Data, Point, Vec2};

use crate::path::{Path, PointId, SplinePoint};

const MIN_CLICK_DISTANCE: f64 = 10.0;
const CLICK_PENALTY: f64 = MIN_CLICK_DISTANCE / 2.0;

#[derive(Clone, Debug, Data)]
pub struct EditSession {
    path: Path,
    paths: Arc<Vec<Path>>,
    selection: Option<PointId>,
}

impl EditSession {
    pub fn new() -> EditSession {
        EditSession {
            path: Path::new(),
            selection: None,
            paths: Arc::new(Vec::new()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_base64_bincode(b64: String) -> Option<EditSession> {
        let bytes = base64::decode(b64.trim_start_matches('?')).ok()?;
        let mut paths: Vec<spline::SplineSpec> = bincode::deserialize(&bytes).ok()?;

        let path = if paths.last().map(|path| !path.is_closed()).unwrap_or(false) {
            Path::from_spline(paths.pop().unwrap())
        } else {
            Path::new()
        };
        let paths = Arc::new(paths.into_iter().map(Path::from_spline).collect());
        let selection = path.last_point().map(|pt| pt.id);
        Some(EditSession {
            path,
            paths,
            selection,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn to_base64_bincode(&self) -> String {
        let paths = self
            .paths
            .iter()
            .chain(Some(&self.path).into_iter())
            .map(Path::solver)
            .collect::<Vec<_>>();
        let bytes = bincode::serialize(&paths).unwrap();
        base64::encode(&bytes)
    }

    pub fn active_path(&self) -> &Path {
        &self.path
    }

    pub fn iter_paths(&self) -> impl Iterator<Item = &Path> {
        Some(&self.path).into_iter().chain(self.paths.iter())
    }

    pub fn bezier(&self) -> BezPath {
        self.iter_paths()
            .flat_map(|p| p.bezier().elements())
            .copied()
            .collect()
    }

    pub fn add_point(&mut self, point: Point, smooth: bool) {
        // if the current path is closed we need to add a new path
        if self.path.is_closed() {
            let path = std::mem::replace(&mut self.path, Path::new());
            Arc::make_mut(&mut self.paths).push(path);
        }

        if self
            .path
            .points()
            .first()
            .map(|pt| pt.point.distance(point) < MIN_CLICK_DISTANCE)
            .unwrap_or(false)
        {
            self.path.close(smooth);
        } else if let Some((idx, _)) = self.nearest_segment_for_point(point) {
            let sel = match idx {
                0 => self.path.insert_point_on_path(point),
                n => Arc::make_mut(&mut self.paths)
                    .get_mut(n - 1)
                    .unwrap()
                    .insert_point_on_path(point),
            };
            self.selection = Some(sel);
        } else {
            let sel = self.path.add_point(point, smooth);
            self.selection = Some(sel);
        }
    }

    pub fn update_for_drag(&mut self, handle: Point) {
        self.path.update_for_drag(handle)
    }

    pub fn delete(&mut self) {
        if let Some(sel) = self.selection.take() {
            self.selection = self.path_containing_pt_mut(sel).delete(sel);
        }
        if self.selection.is_none() {
            Arc::make_mut(&mut self.paths).retain(|path| !path.points().is_empty())
        }
    }

    pub fn nudge_selection(&mut self, delta: Vec2) {
        if let Some(sel) = self.selection {
            self.path_containing_pt_mut(sel).nudge(sel, delta);
        }
    }

    pub fn is_selected(&self, id: PointId) -> bool {
        Some(id) == self.selection
    }

    pub fn selected_point(&self) -> Option<SplinePoint> {
        self.selection.and_then(|id| {
            self.iter_paths()
                .flat_map(Path::points)
                .find(|pt| pt.id == id)
                .copied()
        })
    }

    pub fn set_selection(&mut self, selection: Option<PointId>) {
        self.selection = selection;
    }

    pub fn maybe_convert_line_to_spline(&mut self, point: Point) {
        let closest = self.nearest_segment_for_point(point);
        match closest {
            Some((0, _)) => self
                .path
                .maybe_convert_line_to_spline(point, MIN_CLICK_DISTANCE),
            Some((n, _)) => Arc::make_mut(&mut self.paths)
                .get_mut(n - 1)
                .unwrap()
                .maybe_convert_line_to_spline(point, MIN_CLICK_DISTANCE),
            _ => (),
        }
    }

    /// returns a path index and a distance, where '0' is the active path
    fn nearest_segment_for_point(&self, point: Point) -> Option<(usize, f64)> {
        self.iter_paths().enumerate().fold(None, |acc, (i, path)| {
            let dist = path.nearest_segment_distance(point);
            match acc {
                Some((cur_idx, cur_dist)) if cur_dist < dist => Some((cur_idx, cur_dist)),
                _ if dist < MIN_CLICK_DISTANCE => Some((i, dist)),
                _ => None,
            }
        })
    }

    pub fn toggle_selected_point_type(&mut self) {
        if let Some(id) = self.selection {
            self.path_containing_pt_mut(id).toggle_point_type(id);
        }
    }

    pub fn move_point(&mut self, id: PointId, pos: Point) {
        self.path_containing_pt_mut(id).move_point(id, pos);
    }

    pub fn hit_test_points(&self, point: Point, max_dist: Option<f64>) -> Option<PointId> {
        let max_dist = max_dist.unwrap_or(MIN_CLICK_DISTANCE);
        let mut best = None;
        for p in self.iter_paths().flat_map(Path::points) {
            let dist = p.point.distance(point);
            // penalize the currently selected point
            let sel_penalty = if Some(p.id) == self.selection {
                CLICK_PENALTY
            } else {
                0.0
            };
            let score = dist + sel_penalty;
            if dist < max_dist && best.map(|(s, _id)| score < s).unwrap_or(true) {
                best = Some((score, p.id))
            }
        }
        best.map(|(_score, id)| id)
    }

    pub fn to_json(&self) -> String {
        let paths = [self.path.solver()];
        serde_json::to_string_pretty(&paths).unwrap_or_default()
    }

    fn path_containing_pt_mut(&mut self, point: PointId) -> &mut Path {
        if self.path.contains_point(point) {
            &mut self.path
        } else {
            let paths = Arc::make_mut(&mut self.paths);
            paths.iter_mut().find(|p| p.contains_point(point)).unwrap()
        }
    }
}
