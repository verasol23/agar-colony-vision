"""Reusable image preview Canvas with zoom, pan and optional box drawing."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title, on_box=None):
        super().__init__(parent)
        ttk.Label(self, text=title, font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 3))
        self.canvas = tk.Canvas(self, bg='#182029', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self._image_bgr = None
        self._photo = None
        self._scale = 1.0
        self._origin = (0,0)
        self._zoom = 1.0
        self._pan = (0.0,0.0)
        self._pan_start = None
        self._box_start = None
        self._on_box = on_box
        self.draw_enabled = False
        self.canvas.bind('<Configure>', lambda _e: self.redraw())
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>',lambda _e:self._zoom_step(1.15))
        self.canvas.bind('<Button-5>',lambda _e:self._zoom_step(1.0/1.15))
        self.canvas.bind('<ButtonPress-2>',self._start_pan)
        self.canvas.bind('<B2-Motion>',self._move_pan)
        self.canvas.bind('<ButtonRelease-2>',lambda _e:self._end_pan())
        self.canvas.bind('<ButtonPress-1>',self._start_box)
        self.canvas.bind('<B1-Motion>',self._drag_box)
        self.canvas.bind('<ButtonRelease-1>',self._finish_box)

    def set_image(self, image_bgr, reset_zoom=False):
        self._image_bgr = image_bgr.copy() if image_bgr is not None else None
        if reset_zoom:
            self._zoom = 1.0
            self._pan = (0.0,0.0)
        self.redraw()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = (0.0,0.0)
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        if self._image_bgr is None:
            return
        h,w = self._image_bgr.shape[:2]
        cw,ch = max(self.canvas.winfo_width(),1),max(self.canvas.winfo_height(),1)
        self._scale = max(0.01,min(cw/w,ch/h)*self._zoom)
        tw,th = max(1,int(round(w*self._scale))),max(1,int(round(h*self._scale)))
        ox = (cw-tw)/2 + self._pan[0]
        oy = (ch-th)/2 + self._pan[1]
        self._origin = (ox,oy)
        # Only resize visible source pixels to avoid excessive memory at high zoom.
        x0 = max(0,int((0-ox)/self._scale)); y0 = max(0,int((0-oy)/self._scale))
        x1 = min(w,int((cw-ox)/self._scale)+2); y1 = min(h,int((ch-oy)/self._scale)+2)
        if x1 <= x0 or y1 <= y0:
            return
        cropped = self._image_bgr[y0:y1,x0:x1]
        rgb = cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        rw = max(1,int(round((x1-x0)*self._scale)))
        rh = max(1,int(round((y1-y0)*self._scale)))
        im = im.resize((rw,rh),Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.create_image(ox+x0*self._scale,oy+y0*self._scale,
                                 image=self._photo,anchor='nw')

    def _wheel(self,event):
        self._zoom_step(1.15 if event.delta > 0 else 1/1.15)

    def _zoom_step(self,factor):
        self._zoom = max(1.0,min(5.0,self._zoom*factor))
        self.redraw()

    def _start_pan(self,event):
        self._pan_start = (event.x,event.y)

    def _move_pan(self,event):
        if self._pan_start is None:
            return
        dx,dy=event.x-self._pan_start[0],event.y-self._pan_start[1]
        self._pan=(self._pan[0]+dx,self._pan[1]+dy)
        self._pan_start=(event.x,event.y)
        self.redraw()

    def _end_pan(self):
        self._pan_start = None

    def _to_image(self,x,y):
        if self._image_bgr is None:
            return (0,0)
        h,w = self._image_bgr.shape[:2]
        return (int(max(0,min(w,round((x-self._origin[0])/self._scale)))),
                int(max(0,min(h,round((y-self._origin[1])/self._scale)))))

    def _start_box(self,event):
        if self.draw_enabled and self._on_box and self._image_bgr is not None:
            self._box_start = self._to_image(event.x,event.y)

    def _drag_box(self,event):
        if self._box_start is None:
            return
        self.canvas.delete('preview')
        x0,y0=self._box_start
        x1,y1=self._to_image(event.x,event.y)
        ox,oy=self._origin
        a,b,c,d = (ox+x0*self._scale,oy+y0*self._scale,
                   ox+x1*self._scale,oy+y1*self._scale)
        self.canvas.create_rectangle(a,b,c,d,outline='#ffd54f',width=2,
                                      dash=(4,3),tags='preview')

    def _finish_box(self,event):
        if self._box_start is None:
            return
        start=self._box_start
        end=self._to_image(event.x,event.y)
        self._box_start=None
        self.canvas.delete('preview')
        x1,x2=sorted([start[0],end[0]]); y1,y2=sorted([start[1],end[1]])
        if x2-x1 >= 3 and y2-y1 >= 3:
            self._on_box([x1,y1,x2,y2])
