"""Tkinter GUI: dataset explorer, diagnostic stages, batch runner and manual review."""
from __future__ import annotations
import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2

from .annotations import (read_agar_annotation, save_reviewed_annotations,
                          write_yolo_annotations)
from .batch import run_batch, list_images
from .config import DetectorConfig, COLONY_CLASSES
from .gui_params import ParameterDialog
from .detection import detect_colonies
from .evaluation import summarize
from .imaging import render_detection, render_evaluation, render_stage
from .gui_panels import ImagePanel


class ColonyApp:
    def __init__(self, root):
        self.root = root
        root.title('AGAR Colony Vision — Improved Detector')
        root.geometry('1480x880')
        root.minsize(1120,720)
        self.dataset_dir = None
        self.output_dir = Path.cwd()/'results'
        self.images = []
        self.current_path = None
        self.result = None
        self.gt = []
        self.manual = []
        self.metadata = {}
        self.original_size = None
        self.busy = False
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.stage_var = tk.StringVar(value='Evaluation')
        self.class_var = tk.StringVar(value=COLONY_CLASSES[1])
        self.edit_var = tk.BooleanVar(value=False)
        self.active_config = DetectorConfig()
        self.parameter_dialog = None
        self.config_summary_var = tk.StringVar(value=self._config_summary())
        self.max_var = tk.StringVar(value='')
        self.save_preview_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value='Choose an image or a dataset folder to begin.')
        self.metric_var = tk.StringVar(value='No image evaluated')
        self._build()
        root.after(100,self._drain_events)
        root.bind('<Control-s>',lambda _e:self.save_manual())
        root.bind('<Delete>',lambda _e:self.delete_manual())

    def _build(self):
        header=ttk.Frame(self.root,padding=7)
        header.pack(fill='x')
        ttk.Button(header,text='Open image',command=self.open_image).pack(side='left',padx=3)
        ttk.Button(header,text='Choose dataset folder',command=self.choose_dataset).pack(side='left',padx=3)
        ttk.Button(header,text='Choose output folder',command=self.choose_output).pack(side='left',padx=3)
        ttk.Button(header,text='Run selected',command=self.run_selected).pack(side='left',padx=7)
        ttk.Button(header,text='Run entire folder',command=self.run_folder).pack(side='left',padx=3)
        ttk.Button(header,text='Cancel batch',command=self.cancel_batch).pack(side='left',padx=3)
        self.progress=ttk.Progressbar(header,length=160,mode='determinate')
        self.progress.pack(side='right',padx=5)

        settings=ttk.Frame(self.root,padding=(8,0,8,5))
        settings.pack(fill='x')
        ttk.Label(settings,text='Detector: Improved (single pipeline)').pack(side='left',padx=(0,10))
        ttk.Label(settings,text='Stage').pack(side='left',padx=(10,0))
        self.stage_box=ttk.Combobox(settings,textvariable=self.stage_var,state='readonly',width=22,
                                  values=['Evaluation','Detection','Cropped dish','Dish mask',
                                          'Prepared image','Estimated background','Color distance',
                                          'Hysteresis','Morphology','Watershed labels','Edge mask'])
        self.stage_box.pack(side='left',padx=4)
        self.stage_box.bind('<<ComboboxSelected>>',lambda _e:self.render())
        ttk.Button(settings, text='Parameters...', command=self.open_parameters).pack(side='left', padx=(10, 4))
        ttk.Label(settings,text='Batch limit (blank = all)').pack(side='left',padx=(10,0))
        ttk.Entry(settings,textvariable=self.max_var,width=5).pack(side='left',padx=4)
        ttk.Checkbutton(settings,text='Save preview images',variable=self.save_preview_var).pack(side='left',padx=4)
        ttk.Label(self.root, textvariable=self.config_summary_var, foreground='#475569').pack(
            fill='x', padx=12, pady=(0, 3))

        body=ttk.PanedWindow(self.root,orient='horizontal')
        body.pack(fill='both',expand=True,padx=8,pady=3)
        sidebar=ttk.Frame(body,width=275)
        body.add(sidebar,weight=0)
        ttk.Label(sidebar,text='Dataset images',font=('Arial',11,'bold')).pack(anchor='w',pady=(2,5))
        lst_frame=ttk.Frame(sidebar)
        lst_frame.pack(fill='both',expand=True)
        self.files_list=tk.Listbox(lst_frame,exportselection=False)
        self.files_list.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(lst_frame,orient='vertical',command=self.files_list.yview)
        scroll.pack(side='right',fill='y')
        self.files_list.configure(yscrollcommand=scroll.set)
        self.files_list.bind('<<ListboxSelect>>',self._file_selected)
        ttk.Label(sidebar,text='Review annotations',font=('Arial',11,'bold')).pack(anchor='w',pady=(12,5))
        ttk.Label(sidebar,text='Manual class (not an AI prediction)').pack(anchor='w')
        ttk.Combobox(sidebar,textvariable=self.class_var,values=COLONY_CLASSES,
                     state='readonly',width=23).pack(fill='x',pady=4)
        ttk.Checkbutton(sidebar,text='Draw boxes on left image',variable=self.edit_var,
                        command=self._set_edit_mode).pack(anchor='w')
        self.annotation_list=tk.Listbox(sidebar,height=9,exportselection=False)
        self.annotation_list.pack(fill='x',pady=(4,5))
        self.annotation_list.bind('<<ListboxSelect>>',lambda _e:self.render())
        ttk.Button(sidebar,text='Delete selected annotation',command=self.delete_manual).pack(fill='x',pady=2)
        ttk.Button(sidebar,text='Set selected box class',command=self.change_manual_class).pack(fill='x',pady=2)
        ttk.Button(sidebar,text='Use predictions as editable boxes',command=self.use_pred_as_manual).pack(fill='x',pady=2)
        ttk.Button(sidebar,text='Save reviewed JSON + YOLO',command=self.save_manual).pack(fill='x',pady=2)
        ttk.Button(sidebar,text='Reset zoom / pan',command=self.reset_view).pack(fill='x',pady=2)

        central=ttk.Frame(body)
        body.add(central,weight=1)
        panels=ttk.PanedWindow(central,orient='horizontal')
        panels.pack(fill='both',expand=True)
        self.left=ImagePanel(panels,'Original crop + review annotations',self._add_manual)
        self.right=ImagePanel(panels,'Pipeline stage / detection / TP-FP-FN')
        panels.add(self.left,weight=1)
        panels.add(self.right,weight=1)
        ttk.Label(central,textvariable=self.metric_var,font=('Arial',10,'bold')).pack(fill='x',pady=(6,3))
        ttk.Label(central,text='Zoom: wheel | Pan: middle mouse | Manual box: tick checkbox, then drag on left',
                  foreground='#64748b').pack(anchor='w')
        footer=ttk.Frame(self.root,padding=(8,4,8,8))
        footer.pack(fill='x')
        ttk.Label(footer,textvariable=self.status_var,wraplength=1450).pack(anchor='w')

    def _config_summary(self):
        """Short, always-current description of the applied settings."""
        c = self.active_config
        return (f'Dish={c.dish_shrink:g}% | Gaussian={c.gaussian_sigma:g} | '
                f'Low/High={c.thr_low:g}/{c.thr_high:g} | IoU={c.iou_threshold:.2f}')

    def _config(self):
        """Return the same immutable parameter snapshot for image and batch processing."""
        return self.active_config

    def open_parameters(self):
        """Open one modeless editor without accidentally creating stale duplicates."""
        if self.parameter_dialog is not None and self.parameter_dialog.winfo_exists():
            self.parameter_dialog.lift()
            self.parameter_dialog.focus_force()
            return
        self.parameter_dialog = ParameterDialog(self.root, self.active_config, self.apply_config)

    def apply_config(self, config):
        """Apply validated settings and re-run the selected image when available."""
        if self.busy:
            messagebox.showwarning('Processing in progress',
                                   'Wait until the current image or batch completes before applying a new preset.')
            return False
        self.active_config = config
        self.config_summary_var.set(self._config_summary())
        if self.current_path is not None:
            self._start_selected()
        else:
            self.status_var.set('Parameters applied. Open an image or run a dataset to use them.')
        return True

    def choose_dataset(self):
        if self.busy:return
        path=filedialog.askdirectory(title='Choose countable/ dataset folder')
        if not path:return
        self.dataset_dir=Path(path).resolve()
        self.images=list_images(self.dataset_dir)
        self.files_list.delete(0,'end')
        for image in self.images:
            tag=' [JSON]' if image.with_suffix('.json').is_file() else ' [no JSON]'
            self.files_list.insert('end',image.relative_to(self.dataset_dir).as_posix()+tag)
        self.status_var.set(f'Dataset: {self.dataset_dir} | {len(self.images)} images')
        if self.images:
            self.files_list.selection_set(0)
            self.run_selected()

    def open_image(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Open agar plate',
              filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp'),('All files','*.*')])
        if path:
            self.current_path=Path(path).resolve()
            self._start_selected()

    def choose_output(self):
        if self.busy:return
        path=filedialog.askdirectory(title='Choose a separate results directory')
        if path:
            self.output_dir=Path(path).resolve()
            self.status_var.set(f'Output: {self.output_dir}')

    def _file_selected(self,_event=None):
        if self.busy:return
        selected=self.files_list.curselection()
        if selected and self.images:
            self.current_path=self.images[selected[0]]
            self._start_selected()

    def run_selected(self):
        if self.busy:return
        selected=self.files_list.curselection()
        if selected and self.images:
            self.current_path=self.images[selected[0]]
        if self.current_path is None:
            messagebox.showinfo('Choose image','Open an image or select a dataset image first.')
            return
        self._start_selected()

    def _start_selected(self):
        config=self._config()
        if config is None:return
        self.busy=True
        path=self.current_path
        self.status_var.set('Processing '+str(path.name)+' ...')
        threading.Thread(target=self._selected_worker,args=(path,config),daemon=True).start()

    def _selected_worker(self,path,config):
        try:
            bgr=cv2.imread(str(path))
            if bgr is None:raise OSError(f'Could not read {path}')
            detection=detect_colonies(bgr,config,include_stages=True)
            annotation_path=path.with_suffix('.json')
            try:
                gt=read_agar_annotation(annotation_path,detection['geometry']) or []
                metadata=json.loads(annotation_path.read_text(encoding='utf-8-sig')) if annotation_path.is_file() else {}
                has_gt=annotation_path.is_file()
            except (ValueError,OSError,KeyError,json.JSONDecodeError):
                gt,metadata,has_gt=[],{},False
            payload=(path,detection,gt,metadata,(bgr.shape[1],bgr.shape[0]),has_gt,config)
            self.events.put(('image',payload))
        except Exception as exc:
            self.events.put(('fatal',f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))

    def _load_selected(self,payload):
        self.current_path,self.result,self.gt,self.metadata,self.original_size,self.has_gt,self.current_config=payload
        self.manual=[{'class':entry['class'],'box':list(entry['box'])} for entry in self.gt]
        self._refresh_manual()
        self.busy=False
        self.render(reset=True)
        self.status_var.set(f'Image: {self.current_path.name} | GT: {len(self.gt)} | '
                            f'Colonies: {len(self.result["boxes"])} | '
                            f'Inner: {self.result["inner_count"]} | Edge added: {self.result["edge_added"]}')

    def render(self,reset=False):
        if self.result is None:return
        crop=self.result['crop']
        left_img=crop.copy()
        for n,item in enumerate(self.manual):
            x1,y1,x2,y2=item['box']
            color=(255,0,0)
            if n in self.annotation_list.curselection():color=(0,255,255)
            cv2.rectangle(left_img,(x1,y1),(x2,y2),color,2)
            cv2.putText(left_img,f'{n+1} {item["class"]}',(x1,max(14,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
        self.left.set_image(left_img,reset_zoom=reset)
        boxes=self.result['boxes']
        stage=self.stage_var.get()
        if stage=='Detection':
            right=render_detection(crop,boxes)
        elif stage=='Evaluation':
            truth=[item['box'] for item in self.gt]
            if self.has_gt:
                _stats,match=summarize(boxes,truth,self.current_config.iou_threshold)
                right=render_evaluation(crop,boxes,truth,match)
            else:
                right=render_detection(crop,boxes)
        else:
            right=render_stage(stage,self.result['stages'])
        self.right.set_image(right,reset_zoom=reset)
        if self.has_gt:
            stats,_=summarize(boxes,[it['box'] for it in self.gt],self.current_config.iou_threshold)
            self.metric_var.set(f'Improved | GT={stats["GT"]} Pred={stats["Pred"]} '
                                f'TP={stats["TP"]} FP={stats["FP"]} FN={stats["FN"]} '
                                f'Precision={stats["Precision"]:.3f} Recall={stats["Recall"]:.3f} '
                                f'F1={stats["F1"]:.3f} | IoU={self.current_config.iou_threshold:.2f}')
        else:
            self.metric_var.set(f'Improved | Pred={len(boxes)} | No usable source JSON: metrics unavailable')

    def _set_edit_mode(self):
        self.left.draw_enabled=bool(self.edit_var.get())

    def _refresh_manual(self):
        self.annotation_list.delete(0,'end')
        for i,item in enumerate(self.manual,1):
            self.annotation_list.insert('end',f'#{i} {item["class"]} {item["box"]}')

    def _add_manual(self,box):
        if self.result is None:return
        self.manual.append({'class':self.class_var.get(),'box':box})
        self._refresh_manual()
        self.annotation_list.selection_set('end')
        self.render()
        self.status_var.set('Review box added in memory. Save to export reviewed JSON/YOLO.')

    def delete_manual(self):
        index=self.annotation_list.curselection()
        if not index:return
        del self.manual[index[0]]
        self._refresh_manual();self.render()

    def change_manual_class(self):
        index=self.annotation_list.curselection()
        if not index:return
        self.manual[index[0]]['class']=self.class_var.get()
        self._refresh_manual();self.annotation_list.selection_set(index[0]);self.render()

    def use_pred_as_manual(self):
        if self.result is None:return
        if not messagebox.askyesno('Manual review required',
                'Replace editable boxes with automatic predictions?\n\n'
                'The selected class is a MANUAL label, not an automatic species identification.'):
            return
        cls=self.class_var.get()
        self.manual=[{'class':cls,'box':list(box)}
                     for box in self.result['boxes']]
        self._refresh_manual();self.render()

    def save_manual(self):
        if self.result is None or self.current_path is None:return
        if self.dataset_dir is not None and self.dataset_dir in self.current_path.parents:
            relative=self.current_path.relative_to(self.dataset_dir)
        else:
            relative=Path(self.current_path.name)
        reviewed_json=self.output_dir/'reviewed_labels'/relative.with_suffix('.json')
        reviewed_yolo=self.output_dir/'reviewed_yolo'/relative.with_suffix('.txt')
        reviewed_image=self.output_dir/'reviewed_images'/relative.with_suffix('.png')
        try:
            save_reviewed_annotations(reviewed_json,self.manual,self.result['geometry'],
                                      self.original_size,self.metadata,self.current_path.stem)
            write_yolo_annotations(reviewed_yolo,self.manual)
            reviewed_image.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(reviewed_image),self.result['crop']):
                raise OSError('Failed to save crop image')
        except Exception as e:
            messagebox.showerror('Save failed',str(e));return
        self.status_var.set(f'Saved MANUALLY reviewed labels: {reviewed_json} | YOLO: {reviewed_yolo}')

    def reset_view(self):
        self.left.reset_view();self.right.reset_view()

    def run_folder(self):
        if self.busy:return
        if self.dataset_dir is None:
            messagebox.showinfo('Choose dataset','Choose dataset folder before batch processing.')
            return
        cfg=self._config()
        if cfg is None:return
        limit_txt=self.max_var.get().strip()
        try:
            limit=int(limit_txt) if limit_txt else None
            if limit is not None and limit<0:raise ValueError('Invalid limit')
        except ValueError:
            messagebox.showerror('Invalid limit','Enter a nonnegative integer or leave it blank.')
            return
        if self.dataset_dir == self.output_dir or self.dataset_dir in self.output_dir.parents:
            messagebox.showerror('Unsafe output','Choose an output directory OUTSIDE dataset.')
            return
        self.cancel.clear()
        self.busy=True
        self.progress['value']=0
        self.status_var.set('Batch processing started...')
        threading.Thread(target=self._batch_worker,
                         args=(cfg,limit,bool(self.save_preview_var.get())),daemon=True).start()

    def _batch_worker(self,cfg,limit,save_images):
        try:
            result=run_batch(self.dataset_dir,self.output_dir,cfg,max_images=limit,
                             save_images=save_images,save_evaluations=save_images,
                             progress=lambda n,total,name,status:
                                self.events.put(('progress',(n,total,name,status))),
                             stop_event=self.cancel)
            self.events.put(('batch_done',result))
        except Exception as exc:
            self.events.put(('fatal',f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))

    def cancel_batch(self):
        self.cancel.set()
        if self.busy:self.status_var.set('Stopping after the current image...')

    def _drain_events(self):
        try:
            while True:
                event,payload=self.events.get_nowait()
                if event=='image':self._load_selected(payload)
                elif event=='progress':
                    n,total,name,status=payload
                    self.progress['value']=100*n/max(total,1)
                    self.status_var.set(f'Batch {n}/{total}: {name} — {status}')
                elif event=='batch_done':
                    self.busy=False
                    rows=payload['summary']
                    msg=' | '.join(f'{r["Method"]} F1={r["F1"]:.3f} (TP={r["TP"]}, FP={r["FP"]}, FN={r["FN"]})' for r in rows)
                    self.status_var.set(f'Batch processed {payload["processed"]}/{payload["total"]}; '
                                       f'{len(payload["errors"])} errors; '+msg)
                    if payload['cancelled']:
                        messagebox.showinfo('Batch stopped','Partial results were saved to: '+str(payload['output']))
                    else:
                        messagebox.showinfo('Batch complete',msg+'\n\nCSV and images: '+str(payload['output']))
                elif event=='fatal':
                    self.busy=False
                    self.status_var.set('Error. See details in dialog.')
                    messagebox.showerror('Processing error',payload)
        except queue.Empty:
            pass
        self.root.after(100,self._drain_events)


def main():
    root=tk.Tk()
    ColonyApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
