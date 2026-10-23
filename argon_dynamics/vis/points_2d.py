import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Markers
from vispy.scene.cameras import PanZoomCamera
from functools import partial

class Visual:
  def __init__(self, w: int, h: int, dim: tuple):
    self.canvas = scene.SceneCanvas(
      keys="interactive",
      size=(w, h),
      bgcolor="black",
      show=True,
    )

    self.view = self.canvas.central_widget.add_view()
    self.view.camera = PanZoomCamera(rect=(0, 0, dim[0], dim[1]))
    self.view.camera.aspect = 1.0

    self.markers = Markers(parent=self.view.scene)
    self.points = []
  
  def run(self, positions):
    self.timer = app.Timer(
      interval=1 / 60,
      connect=lambda event: self.render(event, positions),
      start=True,
    )
    app.run()

  def render(self, event, positions):
    frame_idx = event.count % len(positions)
    points = positions[frame_idx]
    self.markers.set_data(
        pos=points,
        face_color="cyan",
        edge_width=0,
        size=10,
    )
    self.canvas.update()