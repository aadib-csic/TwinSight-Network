"""
mesh_utils_final.py — TwinSight Network
Pipeline de visualización corregido para nnUNet Task 08.

Fixes críticos vs versiones anteriores:
  [F1] spacing real pasado a Marching Cubes — elimina las láminas planas
       en CTs anisótropos (Z=5mm). Con spacing=(1,1,1) cada vóxel se
       trataba como cubo perfecto → láminas. Con spacing real → tubos.
  [F2] Transformación affine corregida — cuando MC ya aplica el spacing,
       la affine solo debe aplicar rotación+traslación, no volver a escalar.
  [F3] MIN_VOXELS_TUMOR = 200 — con 686 vóxeles de tumor y umbral 3000
       el tumor se eliminaba completamente. 200 es seguro para nnUNet
       (predicciones más limpias que SwinUNETR).
  [F4] MIN_VOXELS_VESSELS = 100 — con 8808 vóxeles de vasos distribuidos
       en múltiples ramas, umbral 400 eliminaba ramas pequeñas reales.
  [F5] Hígado desde TotalSegmentator (Clase 3 externa) — si se proporciona
       liver_mask, se renderiza como silueta semitransparente de contexto.
  [F6] gaussian_filter sigma adaptativo al spacing — con Z=5mm un sigma=1.2
       isótropo suaviza demasiado en Z. Ahora sigma se escala por eje.
"""

import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────────────────────────

MIN_VOXELS_TUMOR    = 200    # F3: bajado de 3000 — tumor:686 pasaba el filtro
MIN_VOXELS_VESSELS  = 100    # F4: bajado de 400 — preserva ramas pequeñas

BORDER_PAD_FRACTION_XY          = 0.05
BORDER_PAD_FRACTION_Z           = 0.03
BORDER_PAD_MIN                  = 12
BORDER_PAD_FRACTION_XY_VESSELS  = 0.01
BORDER_PAD_FRACTION_Z_VESSELS   = 0.01
BORDER_PAD_MIN_VESSELS          = 5

FOV_EDGE_MARGIN          = 15
FOV_EDGE_MARGIN_VESSELS  = 3
FOV_FILTER_Z_THRESHOLD   = 3.0   # mm — por encima relajamos margen Z
FOV_EDGE_MARGIN_Z_RELAXED = 2

TUMOR_BASE_COLOR  = "#ff1744"
TUMOR_DARK_FACTOR = [0.0, 0.15, 0.25, 0.35, 0.40, 0.45]


# ─────────────────────────────────────────────────────────────────
#  COLOR HELPERS
# ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(r,0,255)), int(np.clip(g,0,255)), int(np.clip(b,0,255)))

def _darken(c, f):
    r,g,b = _hex_to_rgb(c)
    return _rgb_to_hex(r*(1-f), g*(1-f), b*(1-f))

def _lighten(c, f):
    r,g,b = _hex_to_rgb(c)
    return _rgb_to_hex(r+(255-r)*f, g+(255-g)*f, b+(255-b)*f)

def _tumor_color(idx):
    f = TUMOR_DARK_FACTOR[min(idx, len(TUMOR_DARK_FACTOR)-1)]
    return _darken(TUMOR_BASE_COLOR, f)


# ─────────────────────────────────────────────────────────────────
#  HELPERS MORFOLÓGICOS
# ─────────────────────────────────────────────────────────────────

def _adaptive_pad(shape, for_vessels=False):
    if for_vessels:
        px = max(BORDER_PAD_MIN_VESSELS, int(shape[0]*BORDER_PAD_FRACTION_XY_VESSELS))
        py = max(BORDER_PAD_MIN_VESSELS, int(shape[1]*BORDER_PAD_FRACTION_XY_VESSELS))
        pz = max(BORDER_PAD_MIN_VESSELS, int(shape[2]*BORDER_PAD_FRACTION_Z_VESSELS))
    else:
        px = max(BORDER_PAD_MIN, int(shape[0]*BORDER_PAD_FRACTION_XY))
        py = max(BORDER_PAD_MIN, int(shape[1]*BORDER_PAD_FRACTION_XY))
        pz = max(BORDER_PAD_MIN, int(shape[2]*BORDER_PAD_FRACTION_Z))
    return px, py, pz


def _apply_border_padding(mask, for_vessels=False):
    out = mask.copy()
    px, py, pz = _adaptive_pad(mask.shape, for_vessels=for_vessels)
    out[:px,:,:]=0; out[-px:,:,:]=0
    out[:,:py,:]=0; out[:,-py:,:]=0
    out[:,:,:pz]=0; out[:,:,-pz:]=0
    return out


def _filter_fov(labeled, n, shape, margin_xy, margin_z):
    cleaned = np.zeros_like(labeled, dtype=bool)
    for cid in range(1, n+1):
        vox = np.argwhere(labeled == cid)
        if len(vox) == 0:
            continue
        mn = vox.min(axis=0)
        mx = vox.max(axis=0)
        touches = (
            mn[0] < margin_xy or mx[0] > shape[0]-margin_xy or
            mn[1] < margin_xy or mx[1] > shape[1]-margin_xy or
            mn[2] < margin_z  or mx[2] > shape[2]-margin_z
        )
        if not touches:
            cleaned[labeled == cid] = True
    return cleaned


def _clean_mask(mask, min_voxels, apply_fov=True,
                for_vessels=False, spacing_z=1.5):
    padded = _apply_border_padding(mask, for_vessels=for_vessels)
    labeled, n = ndimage.label(padded)
    if n == 0:
        return padded.astype(bool)

    sizes = np.bincount(labeled.ravel())
    filtered = np.zeros_like(padded, dtype=bool)
    for cid in range(1, n+1):
        if sizes[cid] >= min_voxels:
            filtered[labeled == cid] = True

    if not np.any(filtered) or not apply_fov:
        return filtered

    margin_xy = FOV_EDGE_MARGIN_VESSELS if for_vessels else FOV_EDGE_MARGIN
    if for_vessels:
        margin_z = FOV_EDGE_MARGIN_VESSELS
    elif spacing_z > FOV_FILTER_Z_THRESHOLD:
        margin_z = FOV_EDGE_MARGIN_Z_RELAXED
    else:
        margin_z = FOV_EDGE_MARGIN

    labeled2, n2 = ndimage.label(filtered)
    return _filter_fov(labeled2, n2, mask.shape, margin_xy, margin_z)


# ─────────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────────────────────────

def _smooth_and_mesh(binary_mask, sigma_base, affine, spacing=(1.0,1.0,1.0)):
    """
    Marching Cubes con resampleo isótropo previo.

    FIX ANISOTROPIA: Con CTs de spacing Z>>XY (e.g. 0.92x0.92x5mm),
    Marching Cubes sobre vóxeles anisótropos genera superficies en capas
    (aspecto de cebolla) porque los vértices en Z están muy separados.

    Solución: resamplear la máscara a spacing isótropo (ISO_SPACING mm)
    antes de aplicar gaussian_filter y Marching Cubes. Esto garantiza
    superficies esféricas/tubulares suaves independientemente del spacing
    de adquisición del CT.

    El espacio físico se recupera correctamente porque los vértices
    generados en espacio isótropo se transforman al espacio RAS real
    mediante la affine original.
    """
    from scipy.ndimage import zoom as _zoom

    ISO_SPACING = 1.5   # mm — spacing isótropo objetivo

    if np.sum(binary_mask) < 30:
        return None

    sp = np.array(spacing, dtype=float)
    sp = np.where(sp == 0, 1.0, sp)

    # Resamplear a spacing isótropo — elimina el efecto de capas
    scale_factors = tuple(float(s / ISO_SPACING) for s in sp)
    any_aniso = any(abs(s - ISO_SPACING) > 0.1 for s in sp)
    if any_aniso:
        mask_iso = _zoom(binary_mask.astype(np.float32), scale_factors, order=1)
    else:
        mask_iso = binary_mask.astype(np.float32)

    # Gaussian smoothing sobre la máscara isótropa (sigma uniforme en mm)
    sigma_iso = float(sigma_base)   # ya en mm isótropos
    smoothed = gaussian_filter(mask_iso, sigma=sigma_iso)

    if smoothed.max() < 0.5:
        return None

    try:
        # Marching Cubes sobre vóxeles isótropos — spacing=(ISO,ISO,ISO)
        verts, faces, normals, _ = measure.marching_cubes(
            smoothed,
            level=0.5,
            spacing=(ISO_SPACING, ISO_SPACING, ISO_SPACING),
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None

    if len(verts) == 0 or len(faces) == 0:
        return None

    # Transformar al espacio RAS real:
    # Los vértices están en mm isótropos → solo necesitamos rotación + traslación
    rot   = affine[:3, :3]
    scale = np.sqrt((rot**2).sum(axis=0))
    scale = np.where(scale == 0, 1, scale)
    rot_only = rot / scale

    verts_w   = (rot_only @ verts.T).T + affine[:3, 3]
    normals_w = (rot_only @ normals.T).T
    nlen = np.linalg.norm(normals_w, axis=1, keepdims=True)
    normals_w = normals_w / np.where(nlen==0, 1, nlen)

    return (verts_w[:,0], verts_w[:,1], verts_w[:,2],
            faces[:,0],  faces[:,1],  faces[:,2],
            normals_w[:,0], normals_w[:,1], normals_w[:,2])


def _compute_camera(all_x, all_y, all_z):
    if len(all_x) == 0:
        return dict(eye=dict(x=1.5,y=-1.5,z=0.8),
                    center=dict(x=0,y=0,z=0), up=dict(x=0,y=0,z=1))
    dx = float(np.max(all_x)-np.min(all_x))
    dy = float(np.max(all_y)-np.min(all_y))
    dz = float(np.max(all_z)-np.min(all_z))
    diam = max(dx, dy, dz, 1.0)
    zoom = max(1.5, min(3.0, diam/100.0))
    eye  = np.array([1.3, -1.5, 0.75])
    eye  = eye / np.linalg.norm(eye) * zoom
    return dict(eye=dict(x=float(eye[0]),y=float(eye[1]),z=float(eye[2])),
                center=dict(x=0,y=0,z=0), up=dict(x=0,y=0,z=1))


# ─────────────────────────────────────────────────────────────────
#  API PÚBLICA
# ─────────────────────────────────────────────────────────────────

def get_tumor_gps(voxel_data, affine, spacing=(1.5,1.5,1.5),
                  min_voxels=MIN_VOXELS_TUMOR):
    sz = float(spacing[2]) if len(spacing)>2 else 1.5
    clean = _clean_mask((voxel_data==2).astype(bool), min_voxels,
                        apply_fov=True, spacing_z=sz)
    labeled, n = ndimage.label(clean)
    if n == 0:
        return "   - No distinct tumor masses identified after filtering.\n"
    centers = ndimage.center_of_mass(clean, labeled, range(1, n+1))
    sizes   = np.bincount(labeled.ravel())[1:]
    report  = f"   Identified {n} distinct focal lesion(s):\n"
    for i,(c,sz_v) in enumerate(zip(centers, sizes)):
        w = affine @ np.append(c, 1.0)
        report += (f"   T{i+1:02d}  X:{w[0]:+7.1f}  Y:{w[1]:+7.1f}  "
                   f"Z:{w[2]:+7.1f} mm  [{sz_v:,} vox]\n")
    return report


def calculate_volumes(voxel_data, spacing=(1.5,1.5,1.5),
                      min_voxels=MIN_VOXELS_TUMOR):
    vox_cc = np.prod(spacing) / 1000.0
    sz     = float(spacing[2]) if len(spacing)>2 else 1.5

    clean_v = _clean_mask((voxel_data==1).astype(bool), MIN_VOXELS_VESSELS,
                          apply_fov=False, for_vessels=True, spacing_z=sz)
    v_cc = float(np.sum(clean_v)) * vox_cc

    clean_t = _clean_mask((voxel_data==2).astype(bool), min_voxels,
                          apply_fov=True, spacing_z=sz)
    t_cc = float(np.sum(clean_t)) * vox_cc

    body = float(np.sum(voxel_data > 0))
    l_cc = body * vox_cc if body > 0 else max(v_cc, 1.0)
    burden = (t_cc / l_cc * 100.0) if l_cc > 0 else 0.0
    return l_cc, v_cc, t_cc, burden


def generate_plotly_html(voxel_data, affine, output_path,
                         patient_id="", spacing=(1.5,1.5,1.5),
                         liver_mask=None):
    """
    Genera el Digital Twin 3D interactivo.

    Args:
        voxel_data  : array uint8 {0=bg, 1=vessels, 2=tumor}
        affine      : affine RAS 4×4 del NIfTI
        output_path : ruta del HTML de salida
        patient_id  : ID del paciente para el título
        spacing     : (x,y,z) mm/voxel del CT original
        liver_mask  : array bool (mismo shape que voxel_data) de la
                      segmentación hepática de TotalSegmentator.
                      Si None, no se renderiza silueta hepática.
    """
    fig  = go.Figure()
    sz   = float(spacing[2]) if len(spacing)>2 else 1.5
    sp   = tuple(spacing)
    all_x, all_y, all_z = [], [], []

    # ── HÍGADO (TotalSegmentator) ─────────────────────────────
    if liver_mask is not None and np.any(liver_mask):
        result_l = _smooth_and_mesh(liver_mask.astype(bool),
                                    sigma_base=2.0, affine=affine,
                                    spacing=sp)
        if result_l is not None:
            x,y,z,ii,jj,kk,_,_,nz_l = result_l
            all_x.extend(x); all_y.extend(y); all_z.extend(z)
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=ii, j=jj, k=kk,
                name="Liver",
                color="#4a7fb5",
                opacity=0.12,
                flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.4),
                showlegend=True,
                hoverinfo="skip",
            ))

    # ── TUMOR ────────────────────────────────────────────────
    raw_t   = (voxel_data==2).astype(bool)
    clean_t = _clean_mask(raw_t, MIN_VOXELS_TUMOR, apply_fov=True, spacing_z=sz)
    t_markers = clean_t.copy()
    labeled_t, n_t = ndimage.label(clean_t)

    for ci in range(1, n_t+1):
        result = _smooth_and_mesh((labeled_t==ci), sigma_base=1.0,
                                  affine=affine, spacing=sp)
        if result is None:
            continue
        x,y,z,ii,jj,kk,nx,ny,nz = result
        all_x.extend(x); all_y.extend(y); all_z.extend(z)
        bc = _tumor_color(ci-1)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=ii, j=jj, k=kk,
            name="Tumor / Focal Lesion" if ci==1 else f"Lesion {ci}",
            intensity=(nz+1.0)/2.0,
            intensitymode="vertex",
            colorscale=[[0.0,_darken(bc,0.50)],[0.4,_darken(bc,0.15)],
                        [0.7,bc],[1.0,_lighten(bc,0.30)]],
            showscale=False,
            opacity=0.88,
            flatshading=False,
            lighting=dict(ambient=0.30,diffuse=0.75,specular=0.95,
                          roughness=0.20,fresnel=0.60),
            lightposition=dict(x=300,y=-200,z=600),
            showlegend=(ci==1), legendgroup="tumor",
            hoverinfo="skip",
        ))

    # ── VASOS ────────────────────────────────────────────────
    raw_v   = (voxel_data==1).astype(bool)
    clean_v = _clean_mask(raw_v, MIN_VOXELS_VESSELS,
                          apply_fov=False, for_vessels=True, spacing_z=sz)
    result_v = _smooth_and_mesh(clean_v, sigma_base=1.2,
                                affine=affine, spacing=sp)
    if result_v is not None:
        x,y,z,ii,jj,kk,nx,ny,nz = result_v
        all_x.extend(x); all_y.extend(y); all_z.extend(z)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=ii, j=jj, k=kk,
            name="Hepatic Vascular Tree",
            intensity=(nz+1.0)/2.0,
            intensitymode="vertex",
            colorscale=[[0.00,"#001f33"],[0.25,"#004f7a"],
                        [0.55,"#0077b6"],[0.80,"#00b4d8"],[1.00,"#90e0ef"]],
            showscale=False,
            opacity=0.75,
            flatshading=False,
            lighting=dict(ambient=0.25,diffuse=0.80,specular=0.55,
                          roughness=0.35,fresnel=0.35),
            lightposition=dict(x=300,y=-200,z=600),
            showlegend=True,
            hoverinfo="skip",
        ))

    # ── MARCADORES QUIRÚRGICOS ────────────────────────────────
    if n_t > 0 and np.any(t_markers):
        centers  = ndimage.center_of_mass(t_markers, labeled_t, range(1,n_t+1))
        sizes_t  = np.bincount(labeled_t.ravel())[1:]
        vox_mm3  = float(np.prod(spacing))
        xp,yp,zp,hvs,lbs = [],[],[],[],[]

        for i,(c,sv) in enumerate(zip(centers,sizes_t)):
            w  = affine @ np.append(c, 1.0)
            wx,wy,wz = float(w[0]),float(w[1]),float(w[2])
            rm = (3*sv*vox_mm3/(4*np.pi))**(1/3)
            xp.append(wx); yp.append(wy); zp.append(wz)
            lbs.append(f"T{i+1}")
            hvs.append(
                f"<b>Tumor {i+1}</b><br>"
                f"X: {wx:.1f}<br>"
                f"Y: {wy:.1f}<br>"
                f"Z: {wz:.1f} mm"
            )
            fig.add_trace(go.Scatter3d(
                x=[wx],y=[wy],z=[wz], mode="markers",
                marker=dict(size=max(10,rm*0.65),
                            color="rgba(255,220,0,0.07)",
                            symbol="circle",
                            line=dict(color="rgba(255,220,0,0.4)",width=1)),
                showlegend=False, hoverinfo="skip"))

        # Coordenadas visibles directamente en el viewport
        coord_labels = [
            f"<b>{lb}</b>  {wx:+.1f}, {wy:+.1f}, {wz:+.1f} mm"
            for lb, wx, wy, wz in zip(lbs, xp, yp, zp)
        ]

        fig.add_trace(go.Scatter3d(
            x=xp, y=yp, z=zp,
            mode="markers+text",
            name="Surgical Targets",
            marker=dict(size=9, color="#00ffcc", symbol="diamond",
                        line=dict(color="white", width=1.5), opacity=1.0),
            text=coord_labels,
            textposition="top center",
            textfont=dict(family="Courier New, monospace", size=11, color="#00ffcc"),
            hoverinfo="text",
            hovertext=hvs,
            hoverlabel=dict(
                bgcolor="#0a0e14", bordercolor="#00ffcc",
                font=dict(family="Courier New, monospace", size=12, color="#d0e8e8")
            ),
        ))

    # ── LAYOUT ────────────────────────────────────────────────
    camera = _compute_camera(all_x, all_y, all_z)
    pid    = patient_id or "UNKNOWN"
    sz_str = f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm"
    liver_str = "· Liver: TotalSegmentator" if liver_mask is not None else ""

    ax_style = dict(backgroundcolor="#090d14", gridcolor="#111827",
                    zerolinecolor="#192535", showbackground=True,
                    tickfont=dict(size=9,color="#1e3040"))

    fig.update_layout(
        title=dict(
            text=(f"<b>TWINSIGHT NETWORK</b>"
                  f"<span style='color:#1a3a5a'>  ·  </span>"
                  f"<span style='color:#00b4d8'>HEPATIC DIGITAL TWIN</span>"
                  f"<br><span style='font-size:11px;color:#2a4a62'>"
                  f"Patient: {pid}  ·  nnUNet v2 3d_fullres  ·  "
                  f"Spacing: {sz_str}  ·  RAS {liver_str}</span>"),
            x=0.03, y=0.97,
            font=dict(family="Courier New, monospace",size=14,color="#00e5ff"),
        ),
        paper_bgcolor="#060a0f", plot_bgcolor="#060a0f",
        scene=dict(
            xaxis=dict(title=dict(text="X — Right (+) / Left (−)",
                                  font=dict(size=10,color="#2e5070")),
                       color="#2e5070", **ax_style),
            yaxis=dict(title=dict(text="Y — Anterior (+) / Posterior (−)",
                                  font=dict(size=10,color="#2e5070")),
                       color="#2e5070", **ax_style),
            zaxis=dict(title=dict(text="Z — Superior (+) / Inferior (−)",
                                  font=dict(size=10,color="#2e5070")),
                       color="#2e5070", **ax_style),
            aspectmode="data", camera=camera, bgcolor="#060a0f",
        ),
        legend=dict(x=0.01,y=0.01,
                    font=dict(family="Courier New, monospace",size=11,color="#3a6a8a"),
                    bgcolor="rgba(6,10,15,0.90)",
                    bordercolor="#0f1e2e",borderwidth=1,itemsizing="constant"),
        margin=dict(l=0,r=0,b=0,t=80),
        hoverdistance=12, hovermode="closest",
        annotations=[
            dict(text=("<span style='color:#4a7fb5'>●</span> Liver &nbsp;&nbsp;"
                       "<span style='color:#0077b6'>●</span> Vessels &nbsp;&nbsp;"
                       "<span style='color:#ff1744'>●</span> Tumor &nbsp;&nbsp;"
                       "<span style='color:#00ffcc'>◆</span> Targets"),
                 xref="paper",yref="paper",x=0.99,y=0.015,showarrow=False,
                 font=dict(family="Courier New, monospace",size=10,color="#2e5070"),
                 align="right"),
            dict(text="FOR RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS",
                 xref="paper",yref="paper",x=0.5,y=0.002,showarrow=False,
                 font=dict(family="Courier New, monospace",size=9,color="#141e2a"),
                 align="center"),
        ],
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True,
                   config=dict(displayModeBar=True,displaylogo=False,
                               scrollZoom=True))
    print(f"   [Twin saved] {output_path}")
