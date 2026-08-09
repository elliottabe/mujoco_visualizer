"""widget_gui.py — ipywidgets + mediapy interactive GUI for the Visualizer.

Usage in a Jupyter notebook::

    from mujoco_visualizer import Visualizer, WidgetGUI

    viz = Visualizer(xml_path='...', anatomy='my_anatomy.yaml')
    gui = WidgetGUI(viz, qposes=qposes)   # qposes optional
    gui.show()
"""
from __future__ import annotations

import copy
from typing import Optional

import ipywidgets as widgets
import mediapy as media
import mujoco
import numpy as np
from IPython.display import display

from mujoco_visualizer.visualizer import (
    Visualizer,
    _hex_to_rgb,
    _rgb_to_hex,
)


class WidgetGUI:
    """ipywidgets-based interactive GUI for the generic Visualizer.

    Tabs: Pose | Colors | Visualization | Lighting | Camera | Floor | Settings

    The Pose tab is driven by ``viz.anatomy.pose_groups``; if empty, it falls
    back to a flat list of all hinge/slide joints.
    """

    def __init__(self, viz: Visualizer, qposes: Optional[np.ndarray] = None,
                 init_qpos: Optional[np.ndarray] = None):
        self.viz = viz
        self._qposes = qposes
        self._frame_idx = 0
        self._suppress = False

        if init_qpos is not None:
            self._current_qpos = np.array(init_qpos).copy()
        else:
            base_qpos = getattr(viz.model, 'qpos_spring', viz.model.qpos0)
            self._current_qpos = np.array(base_qpos).copy()

        self._BODY_CATEGORIES = viz.anatomy.category_names
        self._ALL_CAMERAS = viz.list_cameras()

        # Build per-category visible geom lists for the Colors tab
        self._geom_names_by_cat: dict = {}
        for cat in self._BODY_CATEGORIES:
            entries = []
            for gid in viz._geom_categories.get(cat, []):
                if viz._orig_geom_rgba[gid, 3] < 0.01:
                    continue
                name = mujoco.mj_id2name(viz.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f'geom_{gid}'
                entries.append((gid, name))
            self._geom_names_by_cat[cat] = entries

        self._out = widgets.Output()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _do_render(self):
        if self._suppress:
            return
        frame = self.viz.render_frame(self._current_qpos, height=480, width=640)
        self._out.clear_output(wait=False)
        with self._out:
            media.show_image(frame)

    # ── Tab: Pose ─────────────────────────────────────────────────────────────

    def _build_tab_pose(self):
        model = self.viz.model
        pose_groups = self.viz.anatomy.pose_groups

        # Build {group_key: [joint_name, ...]} from anatomy.pose_groups, or
        # fall back to one bucket per joint if no pose_groups are configured.
        leg_joints: dict = {}
        leg_groups: dict = {}
        if pose_groups:
            for jid in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if not jname:
                    continue
                jl = jname.lower()
                for grp in pose_groups:
                    if grp.joint_substrings and not any(
                            s.lower() in jl for s in grp.joint_substrings):
                        continue
                    if grp.side_filters:
                        sides = [s for s in grp.side_filters if s.lower() in jl]
                        if not sides:
                            continue
                        key = f'{grp.label}_{sides[0]}'
                    else:
                        key = grp.label
                    leg_joints[jname] = int(model.jnt_qposadr[jid])
                    leg_groups.setdefault(key, []).append(jname)
                    break
        else:
            for jid in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                jtype = int(model.jnt_type[jid])
                if not jname or jtype not in (2, 3):  # SLIDE/HINGE only
                    continue
                leg_joints[jname] = int(model.jnt_qposadr[jid])
                leg_groups.setdefault('All joints', []).append(jname)

        base = np.array(getattr(model, 'qpos_spring', model.qpos0)).copy()
        sliders = {}
        for name, idx in leg_joints.items():
            dflt = float(base[idx])
            sl = widgets.FloatSlider(
                value=dflt,
                min=round(dflt - 1.8, 2), max=round(dflt + 1.8, 2),
                step=0.01,
                description=name[:20],
                continuous_update=False, readout_format='.2f',
                layout=widgets.Layout(width='550px'),
                style={'description_width': '160px'},
            )
            sl._joint_name = name
            sl._joint_idx  = idx
            sliders[name] = sl

        def _on_slider(change):
            self._current_qpos[change['owner']._joint_idx] = change['new']
            self._do_render()

        for sl in sliders.values():
            sl.observe(_on_slider, names='value')

        reset_btn = widgets.Button(description='Reset to springref', button_style='warning')

        def _on_reset(b):
            self._suppress = True
            self._current_qpos[:] = base
            for sl in sliders.values():
                sl.unobserve_all()
                sl.value = float(base[sl._joint_idx])
                sl.observe(_on_slider, names='value')
            self._suppress = False
            self._do_render()

        reset_btn.on_click(_on_reset)

        order = list(leg_groups.keys())
        tab_children = [widgets.VBox([sliders[n] for n in leg_groups[k]]) for k in order]
        leg_tabs = widgets.Tab(children=tab_children)
        for i, k in enumerate(order):
            leg_tabs.set_title(i, k)

        # Rollout playback (only if qposes provided)
        playback_box = self._build_playback_bar()

        return widgets.VBox([reset_btn, leg_tabs, playback_box])

    def _build_playback_bar(self):
        if self._qposes is None:
            return widgets.HBox([widgets.Label('No rollout loaded.')])

        n = len(self._qposes)
        slider = widgets.IntSlider(
            value=0, min=0, max=n - 1, step=1,
            description='Frame', continuous_update=False,
            layout=widgets.Layout(width='500px'),
            style={'description_width': '60px'},
        )
        label = widgets.Label(f'0 / {n}')

        def _on_frame(change):
            self._frame_idx = change['new']
            label.value = f'{self._frame_idx} / {n}'
            self._current_qpos[:] = self._qposes[self._frame_idx]
            self._do_render()

        slider.observe(_on_frame, names='value')
        return widgets.VBox([
            widgets.HTML('<b>Rollout playback</b>'),
            widgets.HBox([slider, label]),
        ])

    # ── Tab: Colors ───────────────────────────────────────────────────────────

    def _build_tab_colors(self):
        _ALL = '— All —'
        self._color_pickers = color_pickers = {}
        geom_rows = {}
        _picker_cbs = {}
        _hex_cbs = {}

        for cat in self._BODY_CATEGORIES:
            entries = self._geom_names_by_cat.get(cat, [])
            hex_val = self.viz.vis_state['colors'].get(cat, '#888888')

            cp = widgets.ColorPicker(
                value=hex_val, description=cat, concise=False,
                layout=widgets.Layout(width='300px'),
                style={'description_width': '110px'},
            )

            def _on_cat(change, cat=cat):
                self.viz.vis_state['colors'][cat] = change['new']
                self._do_render()

            cp.observe(_on_cat, names='value')
            color_pickers[cat] = cp

            dd = widgets.Dropdown(
                options=[_ALL] + [lbl for _, lbl in entries],
                value=_ALL, layout=widgets.Layout(width='210px'),
            )
            gp = widgets.ColorPicker(
                value=hex_val, description='', concise=True,
                layout=widgets.Layout(width='60px', display='none'),
            )
            gt = widgets.Text(
                value=hex_val, placeholder='#rrggbb',
                layout=widgets.Layout(width='86px', display='none'),
            )

            def _on_geom_picker(change, cat=cat, dd=dd, gt=gt):
                sel = dd.value
                if sel == _ALL:
                    return
                gid = next((g for g, lbl in self._geom_names_by_cat[cat] if lbl == sel), None)
                if gid is not None:
                    self.viz.vis_state['geom_colors'][gid] = change['new']
                    gt.unobserve_all()
                    gt.value = change['new']
                    gt.observe(_hex_cbs[cat], names='value')
                    self._do_render()

            def _on_geom_hex(change, cat=cat, dd=dd, gp=gp):
                raw = change['new'].strip()
                if not raw.startswith('#'):
                    raw = '#' + raw
                if len(raw) != 7:
                    return
                try:
                    int(raw[1:], 16)
                except ValueError:
                    return
                sel = dd.value
                if sel == _ALL:
                    return
                gid = next((g for g, lbl in self._geom_names_by_cat[cat] if lbl == sel), None)
                if gid is not None:
                    self.viz.vis_state['geom_colors'][gid] = raw
                    gp.unobserve_all()
                    gp.value = raw
                    gp.observe(_picker_cbs[cat], names='value')
                    self._do_render()

            def _on_dd(change, cat=cat, gp=gp, gt=gt):
                sel = change['new']
                if sel == _ALL:
                    gp.layout.display = 'none'
                    gt.layout.display = 'none'
                else:
                    gid = next((g for g, lbl in self._geom_names_by_cat[cat] if lbl == sel), None)
                    if gid is not None:
                        v = self.viz.vis_state['geom_colors'].get(
                            gid, self.viz.vis_state['colors'][cat])
                        gp.unobserve_all(); gp.value = v
                        gp.observe(_picker_cbs[cat], names='value')
                        gt.unobserve_all(); gt.value = v
                        gt.observe(_hex_cbs[cat], names='value')
                    gp.layout.display = ''
                    gt.layout.display = ''

            _picker_cbs[cat] = _on_geom_picker
            _hex_cbs[cat]    = _on_geom_hex
            dd.observe(_on_dd,          names='value')
            gp.observe(_on_geom_picker, names='value')
            gt.observe(_on_geom_hex,    names='value')
            geom_rows[cat] = widgets.HBox(
                [dd, gp, gt], layout=widgets.Layout(align_items='center'))

        alpha_sl = widgets.FloatSlider(
            value=self.viz.vis_state['alpha'],
            min=0.0, max=1.0, step=0.05,
            description='Global alpha', continuous_update=False,
            readout_format='.2f',
            layout=widgets.Layout(width='420px'),
            style={'description_width': '110px'},
        )

        def _on_alpha(change):
            self.viz.vis_state['alpha'] = change['new']
            self._do_render()

        alpha_sl.observe(_on_alpha, names='value')

        reset_btn = widgets.Button(description='Reset colors', button_style='warning')

        def _on_reset(b):
            self._suppress = True
            self.viz.vis_state['geom_colors'].clear()
            for cat in self._BODY_CATEGORIES:
                dflt = self.viz._cat_default_hex.get(cat, '#888888')
                self.viz.vis_state['colors'][cat] = dflt
                color_pickers[cat].value = dflt
            self.viz.vis_state['alpha'] = 1.0
            alpha_sl.value = 1.0
            self.viz.model.geom_rgba[:] = self.viz._orig_geom_rgba
            self._suppress = False
            self._do_render()

        reset_btn.on_click(_on_reset)

        rows = [widgets.HBox([color_pickers[cat], geom_rows[cat]],
                             layout=widgets.Layout(align_items='center', margin='1px 0'))
                for cat in self._BODY_CATEGORIES]
        return widgets.VBox([
            widgets.HBox([reset_btn, alpha_sl]),
            widgets.HTML('<small style="color:#666">Left: body color. Dropdown: per-geom override.</small>'),
            *rows,
        ])

    # ── Tab: Visualization ────────────────────────────────────────────────────

    def _build_tab_vis(self):
        _FLAGS = [
            ('contact_points', 'Contact Points'), ('contact_forces', 'Contact Forces'),
            ('actuators', 'Actuators'), ('joints', 'Joints'),
            ('transparent', 'Transparent'), ('shadows', 'Shadows'), ('wireframe', 'Wireframe'),
        ]
        checkboxes = {}
        for key, label in _FLAGS:
            cb = widgets.Checkbox(
                value=self.viz.vis_state['vis_flags'].get(key, False),
                description=label, layout=widgets.Layout(width='190px'),
            )
            def _on_cb(change, key=key):
                self.viz.vis_state['vis_flags'][key] = change['new']
                self._do_render()
            cb.observe(_on_cb, names='value')
            checkboxes[key] = cb

        geom_btns = []
        for g in range(6):
            val = self.viz.vis_state['geom_groups'][g]
            btn = widgets.ToggleButton(
                value=val, description=f'Geom {g}',
                button_style='info' if val else '',
                layout=widgets.Layout(width='90px'),
            )
            def _on_geom(change, g=g):
                self.viz.vis_state['geom_groups'][g] = change['new']
                change['owner'].button_style = 'info' if change['new'] else ''
                self._do_render()
            btn.observe(_on_geom, names='value')
            geom_btns.append(btn)

        site_btns = []
        for g in range(6):
            val = self.viz.vis_state['site_groups'][g]
            btn = widgets.ToggleButton(
                value=val, description=f'Site {g}',
                button_style='info' if val else '',
                layout=widgets.Layout(width='90px'),
            )
            def _on_site(change, g=g):
                self.viz.vis_state['site_groups'][g] = change['new']
                change['owner'].button_style = 'info' if change['new'] else ''
                self._do_render()
            btn.observe(_on_site, names='value')
            site_btns.append(btn)

        reset_btn = widgets.Button(description='Reset visualization', button_style='warning')
        _defaults = {
            'vis_flags':  {k: v for k, v in self.viz.vis_state['vis_flags'].items()},
            'geom_groups': self.viz.vis_state['geom_groups'][:],
            'site_groups': self.viz.vis_state['site_groups'][:],
        }

        def _on_reset(b):
            self._suppress = True
            for key, dflt in _defaults['vis_flags'].items():
                self.viz.vis_state['vis_flags'][key] = dflt
                checkboxes[key].value = dflt
            for g, dflt in enumerate(_defaults['geom_groups']):
                self.viz.vis_state['geom_groups'][g] = dflt
                geom_btns[g].value = dflt
                geom_btns[g].button_style = 'info' if dflt else ''
            for g, dflt in enumerate(_defaults['site_groups']):
                self.viz.vis_state['site_groups'][g] = dflt
                site_btns[g].value = dflt
                site_btns[g].button_style = 'info' if dflt else ''
            self._suppress = False
            self._do_render()

        reset_btn.on_click(_on_reset)

        return widgets.VBox([
            widgets.HBox([reset_btn]),
            widgets.HTML('<b>Visualization Flags</b>'),
            widgets.GridBox(
                [checkboxes[k] for k, _ in _FLAGS],
                layout=widgets.Layout(grid_template_columns='repeat(3, 200px)'),
            ),
            widgets.HTML('<b>Geom Groups</b>'),
            widgets.HBox(geom_btns),
            widgets.HTML('<b>Site Groups</b>'),
            widgets.HBox(site_btns),
        ])

    # ── Tab: Lighting ─────────────────────────────────────────────────────────

    def _build_tab_lighting(self):
        CHANNELS = ['R', 'G', 'B']

        def _rgb_sliders(label, init, on_ch):
            sls = []
            for ch in range(3):
                sl = widgets.FloatSlider(
                    value=float(init[ch]), min=0.0, max=1.0, step=0.01,
                    description=f'{label} {CHANNELS[ch]}',
                    continuous_update=False, readout_format='.2f',
                    layout=widgets.Layout(width='420px'),
                    style={'description_width': '100px'},
                )
                def _cb(change, ch=ch): on_ch(ch, change['new'])
                sl.observe(_cb, names='value')
                sls.append(sl)
            return sls

        panels = []
        light_widgets = {}
        lights = self.viz.vis_state['lighting']['lights']
        for li, name in enumerate(['Right', 'Left', 'Tracking']):
            if li >= len(lights):
                break
            ld = lights[li]
            active_cb = widgets.Checkbox(
                value=ld['active'], description='Active',
                layout=widgets.Layout(width='150px'),
            )
            def _on_active(change, li=li):
                self.viz.vis_state['lighting']['lights'][li]['active'] = change['new']
                self._do_render()
            active_cb.observe(_on_active, names='value')

            def _amb(ch, v, li=li):  self.viz.vis_state['lighting']['lights'][li]['ambient'][ch] = v;  self._do_render()
            def _diff(ch, v, li=li): self.viz.vis_state['lighting']['lights'][li]['diffuse'][ch] = v;  self._do_render()
            def _spec(ch, v, li=li): self.viz.vis_state['lighting']['lights'][li]['specular'][ch] = v; self._do_render()
            amb  = _rgb_sliders('Ambient',  ld['ambient'],  _amb)
            diff = _rgb_sliders('Diffuse',  ld['diffuse'],  _diff)
            spec = _rgb_sliders('Specular', ld['specular'], _spec)

            az_sl = widgets.FloatSlider(
                value=ld['dir_az'], min=0.0, max=360.0, step=1.0,
                description='Dir Az', continuous_update=False,
                layout=widgets.Layout(width='420px'),
                style={'description_width': '100px'},
            )
            el_sl = widgets.FloatSlider(
                value=ld['dir_el'], min=-90.0, max=0.0, step=1.0,
                description='Dir El', continuous_update=False,
                layout=widgets.Layout(width='420px'),
                style={'description_width': '100px'},
            )
            def _on_az(change, li=li): self.viz.vis_state['lighting']['lights'][li]['dir_az'] = change['new']; self._do_render()
            def _on_el(change, li=li): self.viz.vis_state['lighting']['lights'][li]['dir_el'] = change['new']; self._do_render()
            az_sl.observe(_on_az, names='value')
            el_sl.observe(_on_el, names='value')

            light_widgets[li] = {'active': active_cb, 'ambient': amb, 'diffuse': diff,
                                  'specular': spec, 'dir_az': az_sl, 'dir_el': el_sl}
            panels.append(widgets.VBox([active_cb] + amb + diff + spec + [az_sl, el_sl]))

        # Headlight
        hl = self.viz.vis_state['lighting']['headlight']
        hl_active = widgets.Checkbox(value=hl['active'], description='Active',
                                     layout=widgets.Layout(width='150px'))
        def _on_hl_active(change):
            self.viz.vis_state['lighting']['headlight']['active'] = change['new']
            self._do_render()
        hl_active.observe(_on_hl_active, names='value')
        def _hl_amb(ch, v):  self.viz.vis_state['lighting']['headlight']['ambient'][ch] = v;  self._do_render()
        def _hl_diff(ch, v): self.viz.vis_state['lighting']['headlight']['diffuse'][ch] = v;  self._do_render()
        def _hl_spec(ch, v): self.viz.vis_state['lighting']['headlight']['specular'][ch] = v; self._do_render()
        hl_amb  = _rgb_sliders('Ambient',  hl['ambient'],  _hl_amb)
        hl_diff = _rgb_sliders('Diffuse',  hl['diffuse'],  _hl_diff)
        hl_spec = _rgb_sliders('Specular', hl['specular'], _hl_spec)
        panels.append(widgets.VBox([hl_active] + hl_amb + hl_diff + hl_spec))
        light_widgets['headlight'] = {'active': hl_active, 'ambient': hl_amb,
                                       'diffuse': hl_diff, 'specular': hl_spec}

        dual_cb = widgets.Checkbox(
            value=self.viz.vis_state['lighting'].get('use_dual_lighting', False),
            description='Dual lighting', layout=widgets.Layout(width='188px'),
        )
        scale_cb = widgets.Checkbox(
            value=self.viz.vis_state['lighting'].get('use_scale_lights', False),
            description='Scale lights', layout=widgets.Layout(width='168px'),
        )
        scale_sl = widgets.FloatSlider(
            value=self.viz.vis_state['lighting'].get('scale_lights_factor', 1.25),
            min=0.1, max=3.0, step=0.05, description='Scale',
            continuous_update=False, readout_format='.2f',
            layout=widgets.Layout(width='300px',
                                  display='' if self.viz.vis_state['lighting'].get('use_scale_lights') else 'none'),
            style={'description_width': '60px'},
        )
        def _on_dual(change):
            self.viz.vis_state['lighting']['use_dual_lighting'] = change['new']
            self._do_render()
        def _on_scale_cb(change):
            self.viz.vis_state['lighting']['use_scale_lights'] = change['new']
            scale_sl.layout.display = '' if change['new'] else 'none'
            self._do_render()
        def _on_scale_sl(change):
            self.viz.vis_state['lighting']['scale_lights_factor'] = change['new']
            self._do_render()
        dual_cb.observe(_on_dual, names='value')
        scale_cb.observe(_on_scale_cb, names='value')
        scale_sl.observe(_on_scale_sl, names='value')

        _init_snap = copy.deepcopy(self.viz.vis_state['lighting'])
        reset_btn = widgets.Button(description='Reset lighting', button_style='warning')

        def _on_reset(b):
            self._suppress = True
            for li, ld in enumerate(_init_snap['lights']):
                self.viz.vis_state['lighting']['lights'][li] = copy.deepcopy(ld)
                if li in light_widgets:
                    lw = light_widgets[li]
                    lw['active'].value = ld['active']
                    for ch in range(3):
                        lw['ambient'][ch].value  = ld['ambient'][ch]
                        lw['diffuse'][ch].value  = ld['diffuse'][ch]
                        lw['specular'][ch].value = ld['specular'][ch]
                    lw['dir_az'].value = ld['dir_az']
                    lw['dir_el'].value = ld['dir_el']
            hl0 = _init_snap['headlight']
            self.viz.vis_state['lighting']['headlight'] = copy.deepcopy(hl0)
            hw = light_widgets['headlight']
            hw['active'].value = hl0['active']
            for ch in range(3):
                hw['ambient'][ch].value  = hl0['ambient'][ch]
                hw['diffuse'][ch].value  = hl0['diffuse'][ch]
                hw['specular'][ch].value = hl0['specular'][ch]
            self.viz.vis_state['lighting']['use_dual_lighting'] = False
            self.viz.vis_state['lighting']['use_scale_lights']  = False
            self.viz.vis_state['lighting']['scale_lights_factor'] = 1.25
            dual_cb.value  = False
            scale_cb.value = False
            scale_sl.value = 1.25
            scale_sl.layout.display = 'none'
            self._suppress = False
            self._do_render()

        reset_btn.on_click(_on_reset)

        acc = widgets.Accordion(children=panels)
        for i, name in enumerate(['Right', 'Left', 'Tracking', 'Headlight'][:len(panels)]):
            acc.set_title(i, name)
        acc.selected_index = None

        return widgets.VBox([
            widgets.HBox([reset_btn]),
            widgets.HBox([dual_cb, scale_cb, scale_sl],
                         layout=widgets.Layout(align_items='center')),
            acc,
        ])

    # ── Tab: Camera ───────────────────────────────────────────────────────────

    def _build_tab_camera(self):
        cam = self.viz.vis_state['camera']

        mode_toggle = widgets.ToggleButtons(
            options=['Named', 'Free orbit'],
            value='Named' if cam.get('mode', 'named') == 'named' else 'Free orbit',
            layout=widgets.Layout(width='300px'),
        )
        cams = list(self._ALL_CAMERAS) or ['<none>']
        cam_value = cam.get('named', '')
        if cam_value not in cams:
            cam_value = cams[0]
        named_dd = widgets.Dropdown(
            options=cams, value=cam_value,
            description='Camera:', layout=widgets.Layout(width='280px'),
        )
        free_type_dd = widgets.Dropdown(
            options=['free', 'fixed', 'track', 'trackcom'],
            value=cam.get('free_type', 'free'),
            description='Cam type:', layout=widgets.Layout(width='320px'),
            style={'description_width': '100px'},
        )
        trackbody_txt = widgets.Text(
            value=cam.get('trackbody', ''), description='Track body:',
            placeholder='e.g. thorax',
            layout=widgets.Layout(width='380px'), style={'description_width': '100px'},
        )
        fixedcam_txt = widgets.Text(
            value=cam.get('fixedcamid', ''), description='Fixed cam:',
            placeholder='e.g. track1',
            layout=widgets.Layout(width='380px'), style={'description_width': '100px'},
        )

        _slider_cfgs = [
            ('azimuth',   'Azimuth',   cam.get('azimuth', 180.0),    0.0,  360.0, 1.0,   '.0f'),
            ('elevation', 'Elevation', cam.get('elevation', -30.0), -90.0,   0.0, 1.0,   '.0f'),
            ('distance',  'Distance',  cam.get('distance', 0.3),     0.02,   2.0, 0.01,  '.3f'),
            ('lookat_x',  'Lookat X',  cam['lookat'][0],            -0.1,    0.1, 0.001, '.3f'),
            ('lookat_y',  'Lookat Y',  cam['lookat'][1],            -0.1,    0.1, 0.001, '.3f'),
            ('lookat_z',  'Lookat Z',  cam['lookat'][2],            -0.1,    0.1, 0.001, '.3f'),
        ]
        free_sliders = {}
        for key, desc, val, mn, mx, step, fmt in _slider_cfgs:
            free_sliders[key] = widgets.FloatSlider(
                value=val, min=mn, max=mx, step=step, description=desc,
                continuous_update=False, readout_format=fmt,
                layout=widgets.Layout(width='420px'),
                style={'description_width': '100px'},
            )

        named_panel = widgets.VBox([named_dd])
        free_panel  = widgets.VBox(
            [free_type_dd, trackbody_txt, fixedcam_txt] +
            list(free_sliders.values())
        )

        if cam.get('mode', 'named') == 'named':
            named_panel.layout.display = ''
            free_panel.layout.display  = 'none'
        else:
            named_panel.layout.display = 'none'
            free_panel.layout.display  = ''

        def _update_free_subwidgets():
            ft = free_type_dd.value
            trackbody_txt.layout.display = '' if ft in ('track', 'trackcom') else 'none'
            fixedcam_txt.layout.display  = '' if ft == 'fixed' else 'none'

        _update_free_subwidgets()

        def _on_mode(change):
            if change['new'] == 'Named':
                self.viz.vis_state['camera']['mode'] = 'named'
                named_panel.layout.display = ''
                free_panel.layout.display  = 'none'
            else:
                self.viz.vis_state['camera']['mode'] = 'free'
                named_panel.layout.display = 'none'
                free_panel.layout.display  = ''
            self._do_render()

        def _on_named(change):
            self.viz.vis_state['camera']['named'] = change['new']
            self._do_render()

        def _on_free_type(change):
            self.viz.vis_state['camera']['free_type'] = change['new']
            _update_free_subwidgets()
            self._do_render()

        def _on_trackbody(change):
            self.viz.vis_state['camera']['trackbody'] = change['new']
            self._do_render()

        def _on_fixedcam(change):
            self.viz.vis_state['camera']['fixedcamid'] = change['new']
            self._do_render()

        def _on_az(change):   self.viz.vis_state['camera']['azimuth']   = change['new']; self._do_render()
        def _on_el(change):   self.viz.vis_state['camera']['elevation']  = change['new']; self._do_render()
        def _on_dist(change): self.viz.vis_state['camera']['distance']  = change['new']; self._do_render()
        def _on_lx(change):   self.viz.vis_state['camera']['lookat'][0] = change['new']; self._do_render()
        def _on_ly(change):   self.viz.vis_state['camera']['lookat'][1] = change['new']; self._do_render()
        def _on_lz(change):   self.viz.vis_state['camera']['lookat'][2] = change['new']; self._do_render()

        mode_toggle.observe(_on_mode,       names='value')
        named_dd.observe(_on_named,         names='value')
        free_type_dd.observe(_on_free_type, names='value')
        trackbody_txt.observe(_on_trackbody, names='value')
        fixedcam_txt.observe(_on_fixedcam,  names='value')
        free_sliders['azimuth'].observe(_on_az,   names='value')
        free_sliders['elevation'].observe(_on_el, names='value')
        free_sliders['distance'].observe(_on_dist, names='value')
        free_sliders['lookat_x'].observe(_on_lx,  names='value')
        free_sliders['lookat_y'].observe(_on_ly,  names='value')
        free_sliders['lookat_z'].observe(_on_lz,  names='value')

        # Camera presets
        preset_name = widgets.Text(
            value='my_view', description='Name:',
            layout=widgets.Layout(width='260px'),
            style={'description_width': '50px'},
        )
        save_preset_btn   = widgets.Button(description='Save camera',   button_style='success', layout=widgets.Layout(width='125px'))
        preset_dd         = widgets.Dropdown(
            options=['(none)'], description='Preset:',
            layout=widgets.Layout(width='260px'),
            style={'description_width': '60px'},
        )
        load_preset_btn   = widgets.Button(description='Load camera',   button_style='info',   layout=widgets.Layout(width='125px'))
        delete_preset_btn = widgets.Button(description='Delete',        button_style='danger', layout=widgets.Layout(width='80px'))
        preset_status     = widgets.HTML('')

        def _refresh_dd():
            preset_dd.options = ['(none)'] + list(self.viz.vis_state.get('camera_presets', {}).keys())

        def _apply_cam_to_widgets(c):
            self._suppress = True
            mode_toggle.value = 'Named' if c.get('mode', 'named') == 'named' else 'Free orbit'
            desired_named = c.get('named', '')
            if desired_named not in named_dd.options:
                desired_named = named_dd.options[0] if named_dd.options else ''
            named_dd.value = desired_named
            free_type_dd.value = c.get('free_type', 'free')
            trackbody_txt.value = c.get('trackbody', '')
            fixedcam_txt.value  = c.get('fixedcamid', '')
            free_sliders['azimuth'].value   = c.get('azimuth', 180.0)
            free_sliders['elevation'].value = c.get('elevation', -30.0)
            free_sliders['distance'].value  = c.get('distance', 0.3)
            free_sliders['lookat_x'].value  = c['lookat'][0]
            free_sliders['lookat_y'].value  = c['lookat'][1]
            free_sliders['lookat_z'].value  = c['lookat'][2]
            self._suppress = False

        def _on_save_preset(b):
            name = preset_name.value.strip()
            if not name:
                preset_status.value = '<span style="color:red">Enter a name</span>'
                return
            self.viz.vis_state.setdefault('camera_presets', {})[name] = copy.deepcopy(
                self.viz.vis_state['camera'])
            _refresh_dd()
            preset_dd.value = name
            preset_status.value = f'<span style="color:green">✓ Saved "{name}"</span>'

        def _on_load_preset(b):
            sel = preset_dd.value
            presets = self.viz.vis_state.get('camera_presets', {})
            if sel == '(none)' or sel not in presets:
                preset_status.value = '<span style="color:orange">Select a preset</span>'
                return
            self.viz.vis_state['camera'] = copy.deepcopy(presets[sel])
            _apply_cam_to_widgets(self.viz.vis_state['camera'])
            self._do_render()
            preset_status.value = f'<span style="color:green">✓ Loaded "{sel}"</span>'

        def _on_delete_preset(b):
            sel = preset_dd.value
            presets = self.viz.vis_state.get('camera_presets', {})
            if sel == '(none)' or sel not in presets:
                return
            del presets[sel]
            _refresh_dd()
            preset_status.value = f'Deleted "{sel}"'

        save_preset_btn.on_click(_on_save_preset)
        load_preset_btn.on_click(_on_load_preset)
        delete_preset_btn.on_click(_on_delete_preset)
        _refresh_dd()

        return widgets.VBox([
            mode_toggle, named_panel, free_panel,
            widgets.HTML('<hr><b>Camera Presets</b>'),
            widgets.HBox([preset_name, save_preset_btn]),
            widgets.HBox([preset_dd, load_preset_btn, delete_preset_btn]),
            preset_status,
        ])

    # ── Tab: Floor ────────────────────────────────────────────────────────────

    def _build_tab_floor(self):
        fld = self.viz.vis_state['floor']
        sky = self.viz.vis_state['skybox']

        floor_color = widgets.ColorPicker(
            value=fld['color'], description='Floor color',
            layout=widgets.Layout(width='300px'),
            style={'description_width': '110px'},
        )
        floor_alpha = widgets.FloatSlider(
            value=fld['alpha'], min=0.0, max=1.0, step=0.05,
            description='Floor alpha', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )
        texrep_x = widgets.FloatSlider(
            value=fld['texrepeat_x'], min=0.1, max=20.0, step=0.1,
            description='Tex repeat X', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )
        texrep_y = widgets.FloatSlider(
            value=fld['texrepeat_y'], min=0.1, max=20.0, step=0.1,
            description='Tex repeat Y', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )
        reflect_sl = widgets.FloatSlider(
            value=fld['reflectance'], min=0.0, max=1.0, step=0.01,
            description='Reflectance', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )
        shine_sl = widgets.FloatSlider(
            value=fld['shininess'], min=0.0, max=1.0, step=0.01,
            description='Shininess', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )
        emis_sl = widgets.FloatSlider(
            value=fld['emission'], min=0.0, max=1.0, step=0.01,
            description='Emission', continuous_update=False,
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
        )

        show_sky  = widgets.Checkbox(value=sky['show'], description='Show skybox',
                                     layout=widgets.Layout(width='180px'))
        sky_top   = widgets.ColorPicker(value=sky['sky_top'], description='Sky top',
                                        layout=widgets.Layout(width='280px'),
                                        style={'description_width': '80px'})
        sky_bot   = widgets.ColorPicker(value=sky['sky_bot'], description='Sky horizon',
                                        layout=widgets.Layout(width='280px'),
                                        style={'description_width': '80px'})

        def _on_floor_color(change): self.viz.vis_state['floor']['color']        = change['new']; self._do_render()
        def _on_floor_alpha(change): self.viz.vis_state['floor']['alpha']         = change['new']; self._do_render()
        def _on_texrep_x(change):    self.viz.vis_state['floor']['texrepeat_x']  = change['new']; self._do_render()
        def _on_texrep_y(change):    self.viz.vis_state['floor']['texrepeat_y']  = change['new']; self._do_render()
        def _on_reflect(change):     self.viz.vis_state['floor']['reflectance']  = change['new']; self._do_render()
        def _on_shine(change):       self.viz.vis_state['floor']['shininess']    = change['new']; self._do_render()
        def _on_emis(change):        self.viz.vis_state['floor']['emission']     = change['new']; self._do_render()
        def _on_show_sky(change):    self.viz.vis_state['skybox']['show']        = change['new']; self._do_render()
        def _on_sky_top(change):     self.viz.vis_state['skybox']['sky_top']     = change['new']; self._do_render()
        def _on_sky_bot(change):     self.viz.vis_state['skybox']['sky_bot']     = change['new']; self._do_render()

        floor_color.observe(_on_floor_color, names='value')
        floor_alpha.observe(_on_floor_alpha, names='value')
        texrep_x.observe(_on_texrep_x,       names='value')
        texrep_y.observe(_on_texrep_y,        names='value')
        reflect_sl.observe(_on_reflect,       names='value')
        shine_sl.observe(_on_shine,           names='value')
        emis_sl.observe(_on_emis,             names='value')
        show_sky.observe(_on_show_sky,        names='value')
        sky_top.observe(_on_sky_top,          names='value')
        sky_bot.observe(_on_sky_bot,          names='value')

        return widgets.VBox([
            widgets.HTML('<b>Floor</b>'),
            floor_color, floor_alpha, texrep_x, texrep_y,
            reflect_sl, shine_sl, emis_sl,
            widgets.HTML('<hr><b>Skybox</b>'),
            show_sky, sky_top, sky_bot,
        ])

    # ── Tab: Settings (save/load JSON) ────────────────────────────────────────

    def _build_tab_settings(self):
        from mujoco_visualizer.render_settings import list_available_settings
        presets = [d['name'] for d in list_available_settings()] or ['Default']
        preset_dd = widgets.Dropdown(
            options=presets, value=presets[0],
            description='Preset:',
            layout=widgets.Layout(width='320px'),
            style={'description_width': '60px'},
        )
        path_txt = widgets.Text(
            value=presets[0], description='Name/Path:',
            layout=widgets.Layout(width='420px'),
            style={'description_width': '80px'},
        )

        def _on_preset_change(change):
            if change['name'] == 'value':
                path_txt.value = change['new']
        preset_dd.observe(_on_preset_change)
        save_btn = widgets.Button(description='Save', button_style='success',
                                  layout=widgets.Layout(width='90px'))
        load_btn = widgets.Button(description='Load', button_style='info',
                                  layout=widgets.Layout(width='90px'))
        status   = widgets.HTML('')

        def _on_save(b):
            try:
                self.viz.save_settings(path_txt.value)
                status.value = f'<span style="color:green">✓ Saved → {path_txt.value}</span>'
            except Exception as e:
                status.value = f'<span style="color:red">Error: {e}</span>'

        def _on_load(b):
            try:
                self.viz.load_settings(path_txt.value)
                self._rebuild_tabs()
                status.value = f'<span style="color:green">✓ Loaded ← {path_txt.value}</span>'
                self._do_render()
            except FileNotFoundError:
                status.value = f'<span style="color:red">Not found: {path_txt.value}</span>'
            except Exception as e:
                status.value = f'<span style="color:red">Error: {e}</span>'

        save_btn.on_click(_on_save)
        load_btn.on_click(_on_load)

        return widgets.VBox([
            widgets.HTML('<b>Save / Load Settings</b>'),
            preset_dd,
            path_txt,
            widgets.HBox([save_btn, load_btn]),
            status,
        ])

    # ── Public: show ──────────────────────────────────────────────────────────

    def _build_all_tabs(self):
        tabs = widgets.Tab(children=[
            self._build_tab_pose(),
            self._build_tab_colors(),
            self._build_tab_vis(),
            self._build_tab_lighting(),
            self._build_tab_camera(),
            self._build_tab_floor(),
            self._build_tab_settings(),
        ])
        for i, name in enumerate(['Pose', 'Colors', 'Visualization',
                                   'Lighting', 'Camera', 'Floor', 'Settings']):
            tabs.set_title(i, name)
        return tabs

    def _rebuild_tabs(self):
        """Rebuild all tab widgets from current vis_state (after load_settings)."""
        if not hasattr(self, '_main_tabs'):
            return
        self._main_tabs.children = self._build_all_tabs().children

    def show(self):
        """Build and display the full widget UI inline in the notebook."""
        self._main_tabs = self._build_all_tabs()
        controls = widgets.VBox([self._main_tabs], layout=widgets.Layout(width='660px'))
        display(widgets.HBox([controls, self._out]))
        self._do_render()
