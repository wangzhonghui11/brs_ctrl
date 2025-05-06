try:
    import open3d as o3d
except ImportError:
    print("Open3D is not installed. Please install it using `pip install open3d`")


class PointCloudVisualizer:
    def __init__(self) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.o3d_pc = o3d.geometry.PointCloud()
        self._initialized = False

    def __call__(self, cloud, rgb):
        self.o3d_pc.points = o3d.utility.Vector3dVector(cloud)
        self.o3d_pc.colors = o3d.utility.Vector3dVector(rgb / 255)
        if not self._initialized:
            self.vis.add_geometry(self.o3d_pc)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
            self._initialized = True
        self.vis.update_geometry(self.o3d_pc)

        opt = self.vis.get_render_option()
        opt.show_coordinate_frame = True

        self.vis.update_renderer()
        self.vis.poll_events()
