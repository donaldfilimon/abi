//! Three-dimensional spatial index with a lazy octree acceleration path.

use std::cmp::Ordering;

const OCTREE_MIN_POINTS: usize = 64;
const OCTREE_LEAF_CAPACITY: usize = 8;
const OCTREE_MAX_DEPTH: u32 = 16;
const BOUNDS_EPSILON: f32 = 1e-3;

/// A point in three-dimensional space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point3D {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
}

/// Supported spatial distance functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMetric {
    /// Straight-line distance.
    Euclidean,
    /// Axis-aligned L1 distance.
    Manhattan,
    /// One minus cosine similarity.
    Cosine,
}

/// One owned index record.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialRecord3D {
    /// Stable record identifier.
    pub id: u32,
    /// Indexed point.
    pub point: Point3D,
    /// Opaque caller payload.
    pub payload: Vec<u8>,
}

/// Borrowed spatial query result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialSearchResult<'index> {
    /// Matching record identifier.
    pub id: u32,
    /// Distance from the query point.
    pub distance: f32,
    /// Matching point.
    pub point: Point3D,
    /// Opaque payload borrowed from the index.
    pub payload: &'index [u8],
}

/// Euclidean distance.
#[must_use]
pub fn euclidean_distance(left: Point3D, right: Point3D) -> f32 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    let dz = left.z - right.z;
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt()
}

/// Manhattan distance.
#[must_use]
pub fn manhattan_distance(left: Point3D, right: Point3D) -> f32 {
    (left.x - right.x).abs() + (left.y - right.y).abs() + (left.z - right.z).abs()
}

/// Cosine distance, returning one when either point has zero norm.
#[must_use]
pub fn cosine_distance(left: Point3D, right: Point3D) -> f32 {
    let dot = left.x * right.x + left.y * right.y + left.z * right.z;
    let left_norm = (left.x * left.x + left.y * left.y + left.z * left.z).sqrt();
    let right_norm = (right.x * right.x + right.y * right.y + right.z * right.z).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        1.0
    } else {
        1.0 - dot / (left_norm * right_norm)
    }
}

/// Calculate distance using the requested metric.
#[must_use]
pub fn calculate_distance(left: Point3D, right: Point3D, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(left, right),
        DistanceMetric::Manhattan => manhattan_distance(left, right),
        DistanceMetric::Cosine => cosine_distance(left, right),
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundingBox {
    min: Point3D,
    max: Point3D,
}

impl BoundingBox {
    fn center(self) -> Point3D {
        Point3D {
            x: self.min.x.midpoint(self.max.x),
            y: self.min.y.midpoint(self.max.y),
            z: self.min.z.midpoint(self.max.z),
        }
    }

    fn min_distance_to(self, point: Point3D, metric: DistanceMetric) -> f32 {
        let dx = 0.0_f32.max((self.min.x - point.x).max(point.x - self.max.x));
        let dy = 0.0_f32.max((self.min.y - point.y).max(point.y - self.max.y));
        let dz = 0.0_f32.max((self.min.z - point.z).max(point.z - self.max.z));
        match metric {
            DistanceMetric::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            DistanceMetric::Manhattan => dx + dy + dz,
            DistanceMetric::Cosine => unreachable!("cosine queries use the linear path"),
        }
    }
}

fn compute_bounds(records: &[SpatialRecord3D]) -> BoundingBox {
    let mut min = records[0].point;
    let mut max = min;
    for record in &records[1..] {
        min.x = min.x.min(record.point.x);
        min.y = min.y.min(record.point.y);
        min.z = min.z.min(record.point.z);
        max.x = max.x.max(record.point.x);
        max.y = max.y.max(record.point.y);
        max.z = max.z.max(record.point.z);
    }
    BoundingBox {
        min: Point3D {
            x: min.x - BOUNDS_EPSILON,
            y: min.y - BOUNDS_EPSILON,
            z: min.z - BOUNDS_EPSILON,
        },
        max: Point3D {
            x: max.x + BOUNDS_EPSILON,
            y: max.y + BOUNDS_EPSILON,
            z: max.z + BOUNDS_EPSILON,
        },
    }
}

fn octant_of(point: Point3D, center: Point3D) -> usize {
    usize::from(point.x >= center.x)
        | (usize::from(point.y >= center.y) << 1)
        | (usize::from(point.z >= center.z) << 2)
}

fn child_bounds(bounds: BoundingBox, center: Point3D, octant: usize) -> BoundingBox {
    let x_high = octant & 1 != 0;
    let y_high = octant & 2 != 0;
    let z_high = octant & 4 != 0;
    BoundingBox {
        min: Point3D {
            x: if x_high { center.x } else { bounds.min.x },
            y: if y_high { center.y } else { bounds.min.y },
            z: if z_high { center.z } else { bounds.min.z },
        },
        max: Point3D {
            x: if x_high { bounds.max.x } else { center.x },
            y: if y_high { bounds.max.y } else { center.y },
            z: if z_high { bounds.max.z } else { center.z },
        },
    }
}

#[derive(Debug)]
struct OctreeNode {
    bounds: BoundingBox,
    children: Option<[usize; 8]>,
    point_indices: Vec<usize>,
}

#[derive(Debug)]
struct Octree {
    nodes: Vec<OctreeNode>,
    root: usize,
}

impl Octree {
    fn build(records: &[SpatialRecord3D]) -> Self {
        let bounds = compute_bounds(records);
        let indices: Vec<usize> = (0..records.len()).collect();
        let mut tree = Self {
            nodes: Vec::new(),
            root: 0,
        };
        tree.root = tree.build_node(records, &indices, bounds, 0);
        tree
    }

    fn build_node(
        &mut self,
        records: &[SpatialRecord3D],
        indices: &[usize],
        bounds: BoundingBox,
        depth: u32,
    ) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(OctreeNode {
            bounds,
            children: None,
            point_indices: Vec::new(),
        });
        if indices.len() <= OCTREE_LEAF_CAPACITY || depth >= OCTREE_MAX_DEPTH {
            self.nodes[node_index]
                .point_indices
                .extend_from_slice(indices);
            return node_index;
        }

        let center = bounds.center();
        let mut buckets: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
        for &index in indices {
            buckets[octant_of(records[index].point, center)].push(index);
        }
        let children = std::array::from_fn(|octant| {
            self.build_node(
                records,
                &buckets[octant],
                child_bounds(bounds, center, octant),
                depth + 1,
            )
        });
        self.nodes[node_index].children = Some(children);
        node_index
    }
}

fn as_result(record: &SpatialRecord3D, distance: f32) -> SpatialSearchResult<'_> {
    SpatialSearchResult {
        id: record.id,
        distance,
        point: record.point,
        payload: &record.payload,
    }
}

fn distance_order(left: &SpatialSearchResult<'_>, right: &SpatialSearchResult<'_>) -> Ordering {
    left.distance.total_cmp(&right.distance)
}

fn radius_walk<'index>(
    tree: &Octree,
    records: &'index [SpatialRecord3D],
    node_index: usize,
    center: Point3D,
    radius: f32,
    metric: DistanceMetric,
    results: &mut Vec<SpatialSearchResult<'index>>,
) {
    let node = &tree.nodes[node_index];
    if node.bounds.min_distance_to(center, metric) > radius {
        return;
    }
    if let Some(children) = node.children {
        for child in children {
            radius_walk(tree, records, child, center, radius, metric, results);
        }
        return;
    }
    for &index in &node.point_indices {
        let record = &records[index];
        let distance = calculate_distance(center, record.point, metric);
        if distance <= radius {
            results.push(as_result(record, distance));
        }
    }
}

fn knn_insert<'index>(
    best: &mut Vec<SpatialSearchResult<'index>>,
    item: SpatialSearchResult<'index>,
    count: usize,
) {
    let position = best
        .iter()
        .position(|current| current.distance > item.distance)
        .unwrap_or(best.len());
    if best.len() < count {
        best.insert(position, item);
    } else if position < count {
        best.insert(position, item);
        best.pop();
    }
}

fn knn_walk<'index>(
    tree: &Octree,
    records: &'index [SpatialRecord3D],
    node_index: usize,
    center: Point3D,
    count: usize,
    metric: DistanceMetric,
    best: &mut Vec<SpatialSearchResult<'index>>,
) {
    let node = &tree.nodes[node_index];
    if best.len() >= count
        && node.bounds.min_distance_to(center, metric) > best[best.len() - 1].distance
    {
        return;
    }
    if let Some(children) = node.children {
        for child in children {
            knn_walk(tree, records, child, center, count, metric, best);
        }
        return;
    }
    for &index in &node.point_indices {
        let record = &records[index];
        let distance = calculate_distance(center, record.point, metric);
        knn_insert(best, as_result(record, distance), count);
    }
}

/// In-memory 3-D spatial index with lazy octree construction.
#[derive(Debug, Default)]
pub struct SpatialIndex3D {
    records: Vec<SpatialRecord3D>,
    octree: Option<Octree>,
    octree_dirty: bool,
}

impl SpatialIndex3D {
    /// Construct an empty index.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
            octree: None,
            octree_dirty: true,
        }
    }

    /// Insert a point and copy its payload.
    pub fn insert(&mut self, id: u32, point: Point3D, payload: &[u8]) {
        self.records.push(SpatialRecord3D {
            id,
            point,
            payload: payload.to_vec(),
        });
        self.octree_dirty = true;
    }

    /// Number of indexed points.
    #[must_use]
    pub fn count(&self) -> usize {
        self.records.len()
    }

    fn ensure_octree(&mut self) {
        if !self.octree_dirty {
            return;
        }
        self.octree =
            (self.records.len() >= OCTREE_MIN_POINTS).then(|| Octree::build(&self.records));
        self.octree_dirty = false;
    }

    /// Find all points within `radius`, closest first.
    pub fn radius_search(
        &mut self,
        center: Point3D,
        radius: f32,
        metric: DistanceMetric,
    ) -> Vec<SpatialSearchResult<'_>> {
        if metric != DistanceMetric::Cosine && self.records.len() >= OCTREE_MIN_POINTS {
            self.ensure_octree();
            let mut results = Vec::new();
            let tree = self.octree.as_ref().expect("threshold builds an octree");
            radius_walk(
                tree,
                &self.records,
                tree.root,
                center,
                radius,
                metric,
                &mut results,
            );
            results.sort_by(distance_order);
            return results;
        }
        let mut results: Vec<_> = self
            .records
            .iter()
            .filter_map(|record| {
                let distance = calculate_distance(center, record.point, metric);
                (distance <= radius).then(|| as_result(record, distance))
            })
            .collect();
        results.sort_by(distance_order);
        results
    }

    /// Return up to `count` nearest points, closest first.
    pub fn nearest_neighbors(
        &mut self,
        center: Point3D,
        count: usize,
        metric: DistanceMetric,
    ) -> Vec<SpatialSearchResult<'_>> {
        if count == 0 {
            return Vec::new();
        }
        if metric != DistanceMetric::Cosine && self.records.len() >= OCTREE_MIN_POINTS {
            self.ensure_octree();
            let mut best = Vec::with_capacity(count.min(self.records.len()));
            let tree = self.octree.as_ref().expect("threshold builds an octree");
            knn_walk(
                tree,
                &self.records,
                tree.root,
                center,
                count,
                metric,
                &mut best,
            );
            return best;
        }
        let mut results: Vec<_> = self
            .records
            .iter()
            .map(|record| as_result(record, calculate_distance(center, record.point, metric)))
            .collect();
        results.sort_by(distance_order);
        results.truncate(count);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_radius_and_nearest_neighbor_match_the_oracle_example() {
        let mut index = SpatialIndex3D::new();
        index.insert(1, Point3D::default(), b"Origin");
        index.insert(
            2,
            Point3D {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            b"Unit X",
        );
        index.insert(
            3,
            Point3D {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
            b"Y Coordinate 2",
        );
        index.insert(
            4,
            Point3D {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            b"Diagonal Node",
        );
        assert_eq!(index.count(), 4);
        let radius: Vec<_> = index
            .radius_search(Point3D::default(), 1.5, DistanceMetric::Euclidean)
            .iter()
            .map(|result| result.id)
            .collect();
        assert_eq!(radius, [1, 2]);
        let nearest: Vec<_> = index
            .nearest_neighbors(Point3D::default(), 2, DistanceMetric::Euclidean)
            .iter()
            .map(|result| result.id)
            .collect();
        assert_eq!(nearest, [1, 2]);
    }

    #[test]
    fn distance_metrics_match_the_oracle_values() {
        let left = Point3D {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let right = Point3D {
            x: 4.0,
            y: 6.0,
            z: 8.0,
        };
        assert!((euclidean_distance(left, right) - 7.071_068).abs() < 0.0001);
        assert!((manhattan_distance(left, right) - 12.0).abs() < f32::EPSILON);
        assert!((cosine_distance(left, right) - 0.007_416_7).abs() < 0.0001);
        assert!((cosine_distance(Point3D::default(), right) - 1.0).abs() < f32::EPSILON);
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let upper = u16::try_from(self.0 >> 48).expect("upper bits fit");
            f32::from(upper) / f32::from(u16::MAX)
        }
    }

    fn linear_radius_ids(
        records: &[SpatialRecord3D],
        center: Point3D,
        radius: f32,
        metric: DistanceMetric,
    ) -> Vec<u32> {
        let mut results: Vec<_> = records
            .iter()
            .filter_map(|record| {
                let distance = calculate_distance(center, record.point, metric);
                (distance <= radius).then_some((record.id, distance))
            })
            .collect();
        results.sort_by(|left, right| left.1.total_cmp(&right.1));
        results
            .into_iter()
            .map(|(identifier, _)| identifier)
            .collect()
    }

    fn linear_knn_ids(
        records: &[SpatialRecord3D],
        center: Point3D,
        count: usize,
        metric: DistanceMetric,
    ) -> Vec<u32> {
        let mut results: Vec<_> = records
            .iter()
            .map(|record| (record.id, calculate_distance(center, record.point, metric)))
            .collect();
        results.sort_by(|left, right| left.1.total_cmp(&right.1));
        results.truncate(count);
        results
            .into_iter()
            .map(|(identifier, _)| identifier)
            .collect()
    }

    #[test]
    fn octree_matches_linear_scan_across_threshold_and_distributions() {
        let mut random = Lcg(1234);
        for size in [10_usize, 63, 64, 65, 200] {
            for clustered in [false, true] {
                let mut index = SpatialIndex3D::new();
                for identifier in 0..size {
                    let cluster = if clustered {
                        [-50.0, 0.0, 50.0][identifier % 3]
                    } else {
                        0.0
                    };
                    let spread = if clustered { 2.0 } else { 200.0 };
                    let offset = if clustered { -1.0 } else { -100.0 };
                    let point = Point3D {
                        x: cluster + random.next() * spread + offset,
                        y: cluster + random.next() * spread + offset,
                        z: cluster + random.next() * spread + offset,
                    };
                    index.insert(u32::try_from(identifier).expect("test id"), point, b"");
                }
                let center = Point3D {
                    x: random.next() * 200.0 - 100.0,
                    y: random.next() * 200.0 - 100.0,
                    z: random.next() * 200.0 - 100.0,
                };
                for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
                    let expected_radius = linear_radius_ids(&index.records, center, 40.0, metric);
                    let actual_radius: Vec<_> = index
                        .radius_search(center, 40.0, metric)
                        .iter()
                        .map(|result| result.id)
                        .collect();
                    assert_eq!(actual_radius, expected_radius);

                    let expected_knn = linear_knn_ids(&index.records, center, 5, metric);
                    let actual_knn: Vec<_> = index
                        .nearest_neighbors(center, 5, metric)
                        .iter()
                        .map(|result| result.id)
                        .collect();
                    assert_eq!(actual_knn, expected_knn);
                }
            }
        }
    }

    #[test]
    fn insert_after_first_octree_build_invalidates_and_rebuilds() {
        let mut index = SpatialIndex3D::new();
        for identifier in 0_u16..64 {
            index.insert(
                u32::from(identifier),
                Point3D {
                    x: f32::from(identifier),
                    y: 0.0,
                    z: 0.0,
                },
                b"",
            );
        }
        assert_eq!(
            index.nearest_neighbors(Point3D::default(), 1, DistanceMetric::Euclidean)[0].id,
            0
        );
        index.insert(100, Point3D::default(), b"new");
        let nearest = index.nearest_neighbors(Point3D::default(), 2, DistanceMetric::Euclidean);
        assert_eq!(nearest.len(), 2);
        assert!(nearest.iter().any(|result| result.id == 100));
    }
}
