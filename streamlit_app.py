import numpy as np
import streamlit as st
import torch, detectron2
import os, json, cv2, random, yaml
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt
from streamlit_image_comparison import image_comparison
from shapely.geometry import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances

# DATA_TYPE = "Only_complex"
DATA_TYPE = "Only_easily"
# DATA_TYPE = "Merged"
MODEL_PATH = f"./Models/{DATA_TYPE}"

if "predictor" not in st.session_state:
    cfg = get_cfg()
    cfg.merge_from_file(f"{MODEL_PATH}/config.yaml")
    cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    current_time = datetime.now().strftime('%H:%M:%S')
    register_coco_instances(f"custom_test_dataset_{current_time}", {}, f"./Datasets/{DATA_TYPE}/test.json", f"./Datasets/{DATA_TYPE}")
    class_names = ["Nucleus", "Cytoplasm"]
    MetadataCatalog.get(f"custom_test_dataset_{current_time}").thing_classes = class_names
    cfg.DATASETS.TEST = (f"custom_test_dataset_{current_time}",)

    metadata = MetadataCatalog.get(f"custom_test_dataset_{current_time}")

    st.session_state.predictor = predictor
    st.session_state.cfg = cfg
    st.session_state.metadata = metadata



uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.image_test = image
    if uploaded_file.name not in st.session_state:
        st.session_state[uploaded_file.name] = uploaded_file.name
        if "vis_image" in st.session_state:
            del st.session_state["vis_image"]
            del st.session_state["selected_image"]
        

    if "vis_image" not in st.session_state:
#########################################################################################################################################################################################
    # รับเดต้าเข้ามาและประมวลโดยโมเดล
        image_np = np.array(image)
        image_np = image_np.astype(np.float32)
        if image_np.max() > 1.0:
            image_np = image_np/255.0
            image_np = (image_np*255).astype(np.uint8)
        image_copy = image_np.copy()

        import base64
        from io import BytesIO
        buffered = BytesIO()

        polygons = []
        st.session_state.image_test.save(buffered, format="JPEG")
        st.session_state.img_str = base64.b64encode(buffered.getvalue()).decode()

        st.session_state.image_original = image_np

        outputs = st.session_state.predictor(image_np)

        v = Visualizer(image_np[:, :, ::-1], metadata=MetadataCatalog.get(st.session_state.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_image = out.get_image()[:, :, ::-1]

        ###################################### Mask ทั้งหมด
        st.session_state.vis_image = vis_image
        ######################################
        
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        st.session_state.count_cyto = []
        st.session_state.count_nucl = []

        for i in range(len(pred_classes)):
            class_id = pred_classes[i]
            class_name = st.session_state.metadata.get("thing_classes", [])[class_id] if st.session_state.metadata.get("thing_classes", []) else str(class_id)
            box = pred_boxes[i]
            score = scores[i]

        if instances.has("pred_masks"):
            pred_masks = instances.pred_masks.numpy()

            for i, mask in enumerate(pred_masks):
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if instances.has("pred_masks"):
            pred_masks = instances.pred_masks.numpy()
            for i, mask in enumerate(pred_masks):
                # print(i)
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    points = [(int(x), int(y)) for [[x, y]] in contour]
                    label_id = int(instances.pred_classes[i])
                    label_text = f"Nucleus" if label_id == 0 else "Cytoplasm"
                    area = cv2.contourArea(contour)
                    if label_id == 0:
                        st.session_state.count_nucl.append(area)
                    else:
                        st.session_state.count_cyto.append(area)
                    polygons.append({
                        "points": points,
                        "label": label_text
                    })
        def label_to_class(label):
            return label.lower().replace(" ", "-")

        # Separate polygons by label
        st.session_state.nucleus_polygons = []
        st.session_state.other_polygons = []

        for poly in polygons:
            if poly["label"].lower() == "nucleus":
                st.session_state.nucleus_polygons.append(poly)
            else:
                st.session_state.other_polygons.append(poly)

        # First render non-nucleus polygons (e.g., cytoplasm)
        st.session_state.polygon_html = ""
        for poly in st.session_state.other_polygons + st.session_state.nucleus_polygons:  # order: others first, nucleus last
            points_str = " ".join(f"{x*67/100},{y*67/100}" for x, y in poly["points"])
            tooltip = poly["label"]
            class_name = label_to_class(tooltip)
            st.session_state.polygon_html += f'''
            <polygon points="{points_str}" class="hoverable {class_name}" data-label="{tooltip}"></polygon>
            '''

        st.session_state.html_code = f"""
        <style>
        .hoverable {{
            stroke-width: 2;
            transition: fill 0.2s ease, stroke 0.2s ease;
        }}

        /* Default colors */
        .nucleus {{
            fill: rgba(255, 0, 0, 0.01);
            stroke: rgba(255, 0, 0, 0.01);;
        }}
        .cytoplasm {{
            fill: rgba(0, 255, 0, 0.01);
            stroke: rgba(0, 255, 0, 0.01);;
        }}

        /* Hover colors */
        .nucleus:hover {{
            fill: rgba(255, 0, 0, 0.2);
            stroke: rgba(255, 0, 0, 0.5);;
        }}
        .cytoplasm:hover {{
            fill: rgba(0, 255, 0, 0.2);
            stroke: rgba(0, 255, 0, 0.5);
        }}

        .tooltip {{
            position: absolute;
            background-color: black;
            color: white;
            padding: 3px 6px;
            border-radius: 3px;
            font-size: 12px;
            display: none;
            pointer-events: none;
            z-index: 10;
        }}
        .svg-container {{
            position: relative;
            display: inline-block;
            width: fit-content;
            height: fit-content;
            overflow: hidden;
            line-height: 0;
            margin: 0;
            padding: 0;
        }}
        </style>

        <div class="svg-container">
            <img src="data:image/png;base64,{st.session_state.img_str}" style="width: 100%; display: block;" id="mainImage"/>
            <svg style="position:absolute; top:0; left:0; width:100%; height:100%;" id="overlaySvg">
                {st.session_state.polygon_html}
            </svg>
            <div class="tooltip" id="tooltip"></div>
        </div>

        <script>
        const tooltip = document.getElementById("tooltip");
        const svg = document.getElementById("overlaySvg");

        svg.addEventListener("mousemove", function(e) {{
            const target = e.target;
            if (target.tagName === "polygon") {{
                const label = target.getAttribute("data-label");
                tooltip.textContent = label;
                tooltip.style.left = (e.pageX)-10 + "px";
                tooltip.style.top = (e.pageY)-10 + "px";
                tooltip.style.display = "block";
            }} else {{
                tooltip.style.display = "none";
            }}
        }});
        </script>
        """
        
        
#########################################################################################################################################################################################
    # Check เรื่อยเปื่อย
        def get_random_color():
            return tuple([random.randint(0, 255) for _ in range(3)])

        def get_centroid(mask):
            M = cv2.moments((mask * 255).astype(np.uint8))
            if M["m00"] == 0:
                return None
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        image__ = image_np.copy()

        nucleus_indices = [i for i, cls in enumerate(pred_classes) if cls == 0]  # Nucleus
        cytoplasm_indices = [i for i, cls in enumerate(pred_classes) if cls == 1]  # Cytoplasm

        valid_pairs = []
        for nuc_idx in nucleus_indices:
            nuc_mask = pred_masks[nuc_idx]
            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_nuc:
                continue
            try:
                nuc_poly = Polygon(contours_nuc[0].squeeze())
            except:
                continue

            for cyt_idx in cytoplasm_indices:
                cyt_mask = pred_masks[cyt_idx]
                contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours_cyt:
                    continue
                try:
                    cyt_poly = Polygon(contours_cyt[0].squeeze())
                except:
                    continue

                if nuc_poly.within(cyt_poly):
                    valid_pairs.append((nuc_idx, cyt_idx))


        scored_pairs = []
        for nuc_idx, cyt_idx in valid_pairs:
            nuc_centroid = get_centroid(pred_masks[nuc_idx])
            cyt_centroid = get_centroid(pred_masks[cyt_idx])
            if nuc_centroid is None or cyt_centroid is None:
                continue
            distance = np.linalg.norm(np.array(nuc_centroid) - np.array(cyt_centroid))
            scored_pairs.append((nuc_idx, cyt_idx, distance))

        scored_pairs.sort(key=lambda x: x[2])

        matched_nuclei = set()
        matched_cytoplasms = set()
        st.session_state.matched_pairs = []

        for nuc_idx, cyt_idx, _ in scored_pairs:
            if nuc_idx not in matched_nuclei and cyt_idx not in matched_cytoplasms:
                st.session_state.matched_pairs.append((nuc_idx, cyt_idx))
                matched_nuclei.add(nuc_idx)
                matched_cytoplasms.add(cyt_idx)

        for label_idx, (nuc_idx, cyt_idx) in enumerate(st.session_state.matched_pairs, start=1):
            nuc_mask = pred_masks[nuc_idx]
            cyt_mask = pred_masks[cyt_idx]

            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color = get_random_color()
            cv2.drawContours(image__, contours_nuc, -1, color, 2)
            cv2.drawContours(image__, contours_cyt, -1, color, 2)

            M_nuc = cv2.moments((nuc_mask * 255).astype(np.uint8))
            cX_nuc = int(M_nuc["m10"] / M_nuc["m00"]) if M_nuc["m00"] != 0 else 0
            cY_nuc = int(M_nuc["m01"] / M_nuc["m00"]) if M_nuc["m00"] != 0 else 0

            M_cyt = cv2.moments((cyt_mask * 255).astype(np.uint8))
            cX_cyt = int(M_cyt["m10"] / M_cyt["m00"]) if M_cyt["m00"] != 0 else 0
            cY_cyt = int(M_cyt["m01"] / M_cyt["m00"]) if M_cyt["m00"] != 0 else 0

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # cv2.putText(image__, f"Nucleus {label_idx}", (cX_nuc, cY_nuc),
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            # cv2.putText(image__, f"Cytoplasm {label_idx}", (cX_cyt, cY_cyt),
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            
        ######################################## ไม่ได้ใช้แล้วลืมด้วยคืออะไร
        st.session_state.image_nuclei = image__
        ########################################




#########################################################################################################################################################################################
    # จับคู่ Nucleus, Cytoplasm
        st.session_state.nc_ratio_ = []
        st.session_state.pairs_nuc = []
        st.session_state.pairs_cyt = []
        def get_gradient_color(ratio):
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colormap = cm.get_cmap("RdYlGn_r")  # Green to Red
            rgba = colormap(norm(ratio))  # Get RGBA
            rgb = tuple(int(c * 255) for c in rgba[:3])
            return rgb  # Convert RGB to BGR for OpenCV

        image_display = image_np.copy()
        test_rst = []
        for label_idx, (nuc_idx, cyt_idx) in enumerate(st.session_state.matched_pairs, start=1):
            nuc_mask = pred_masks[nuc_idx]
            cyt_mask = pred_masks[cyt_idx]

            # Calculate area ratio
            area_nuc = np.sum(nuc_mask)
            area_cyt = np.sum(cyt_mask)
            nc_ratio = area_nuc / area_cyt if area_cyt != 0 else 0
            # print(nc_ratio, area_nuc, area_cyt)

            st.session_state.pairs_nuc.append(area_nuc)
            st.session_state.pairs_cyt.append(area_cyt)
            st.session_state.nc_ratio_.append(nc_ratio)

            # Get color based on area ratio
            color = get_gradient_color(nc_ratio)

            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            test_rst.append((nuc_idx, cyt_idx, nc_ratio))

            # Draw contours with the gradient color
            cv2.drawContours(image_display, contours_nuc, -1, color, 2)
            cv2.drawContours(image_display, contours_cyt, -1, color, 2)

            # Get centroids
            cX_nuc, cY_nuc = get_centroid(nuc_mask)
            cX_cyt, cY_cyt = get_centroid(cyt_mask)

            # Optional info
            distance = np.linalg.norm(np.array([cX_nuc, cY_nuc]) - np.array([cX_cyt, cY_cyt]))
            info_text = f"#{label_idx} nc:{nc_ratio:.2f}"
            mid_point = ((cX_nuc + cX_cyt) // 2 , (cY_nuc + cY_cyt) // 2)

            # Add info text on the image
            # cv2.putText(image_display, info_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # print(st.session_state.pairs)
        # Show final plot
        ############################################## Matched
        st.session_state.image_display = image_display
        ##############################################


#########################################################################################################################################################################################
    # หา unmatched ต่อ
        # Draw unmatched nuclei (nuclei not inside any cytoplasm)
        image_unmatched = image_np.copy()

        # Get indices of nuclei not inside any cytoplasm
        unmatched_nuclei = []

        for nuc_idx in nucleus_indices:
            nuc_mask = pred_masks[nuc_idx]
            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_nuc:
                try:
                    nuc_poly = Polygon(contours_nuc[0].squeeze())  # Get the nucleus polygon
                except:
                    continue

                # Check if nucleus is not inside any cytoplasm
                inside_any_cytoplasm = False
                for cyt_idx in cytoplasm_indices:
                    cyt_mask = pred_masks[cyt_idx]
                    contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_cyt:
                        try:
                            cyt_poly = Polygon(contours_cyt[0].squeeze())  # Get the cytoplasm polygon
                        except:
                            continue

                        # If nucleus is within cytoplasm, mark it as inside
                        if nuc_poly.within(cyt_poly):
                            inside_any_cytoplasm = True
                            break  # No need to check other cytoplasms

                # If nucleus is not inside any cytoplasm, add to unmatched nuclei
                if not inside_any_cytoplasm:
                    unmatched_nuclei.append(nuc_idx)

        # Now, draw and label the unmatched nuclei
        for idx in unmatched_nuclei:
            mask = pred_masks[idx]
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                color = (255, 0, 0)  # Blue for unmatched nuclei
                cv2.drawContours(image_unmatched, contours, -1, color, 2)
                centroid = get_centroid(mask)
                # if centroid:
                #     cv2.putText(image_unmatched, f"N{idx}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ################################################## Unmatched
        st.session_state.image_unmatched = image_unmatched
        ##################################################



#########################################################################################################################################################################################
    # รวมระหว่าง Matched, Unmatched(ไม่ปกติ)
        def get_gradient_color(ratio):
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colormap = cm.get_cmap("RdYlGn_r")  # Green to Red
            rgba = colormap(norm(ratio))  # Get RGBA
            rgb = tuple(int(c * 255) for c in rgba[:3])
            return rgb  # RGB for OpenCV

        # Create a single image to draw everything on
        image_combined = image_np.copy()

        # === 1. Draw matched nucleus–cytoplasm pairs ===
        test_rst = []
        for label_idx, (nuc_idx, cyt_idx) in enumerate(st.session_state.matched_pairs, start=1):
            nuc_mask = pred_masks[nuc_idx]
            cyt_mask = pred_masks[cyt_idx]

            area_nuc = np.sum(nuc_mask)
            area_cyt = np.sum(cyt_mask)
            nc_ratio = area_nuc / area_cyt if area_cyt != 0 else 0

            color = get_gradient_color(nc_ratio)

            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            test_rst.append((nuc_idx, cyt_idx, nc_ratio))

            # cv2.drawContours(image_combined, contours_nuc, -1, color, 2)
            # cv2.drawContours(image_combined, contours_cyt, -1, color, 2)

            c_nuc = get_centroid(nuc_mask)
            c_cyt = get_centroid(cyt_mask)
            if c_nuc and c_cyt:
                cX_nuc, cY_nuc = c_nuc
                cX_cyt, cY_cyt = c_cyt
                info_text = f"#{label_idx} nc:{nc_ratio:.2f}"
                mid_point = ((cX_nuc + cX_cyt) // 2 , (cY_nuc + cY_cyt) // 2)
                # cv2.putText(image_combined, info_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # === 2. Draw unmatched nuclei ===
        unmatched_nuclei = []
        for nuc_idx in nucleus_indices:
            nuc_mask = pred_masks[nuc_idx]
            contours_nuc, _ = cv2.findContours((nuc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_nuc:
                try:
                    nuc_poly = Polygon(contours_nuc[0].squeeze())
                except:
                    continue

                inside_any_cytoplasm = False
                for cyt_idx in cytoplasm_indices:
                    cyt_mask = pred_masks[cyt_idx]
                    contours_cyt, _ = cv2.findContours((cyt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_cyt:
                        try:
                            cyt_poly = Polygon(contours_cyt[0].squeeze())
                        except:
                            continue

                        if nuc_poly.within(cyt_poly):
                            inside_any_cytoplasm = True
                            break

                if not inside_any_cytoplasm:
                    unmatched_nuclei.append(nuc_idx)

        # Draw unmatched nuclei in blue
        for idx in unmatched_nuclei:
            mask = pred_masks[idx]
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                color = (255, 0, 0)  # Blue
                # cv2.drawContours(image_combined, contours, -1, color, 2)
                centroid = get_centroid(mask)

        ################################################ Combined
        st.session_state.image_combined = image_combined
        ################################################

        import base64, io, colorsys

        # ---------- helpers ---------------------------------------------------------
        def rgba_str(rgb_tuple, alpha):
            r, g, b = rgb_tuple
            return f"rgba({r},{g},{b},{alpha})"

        def rgb_to_css2(rgb_tuple2):
            return f"rgb{rgb_tuple2}"

        def poly_points_str2(mask2, downscale2=0.67):
            cnts2, _ = cv2.findContours((mask2*255).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts2:
                return ""
            pts2 = cnts2[0].squeeze()
            return " ".join(f"{int(x2*downscale2)},{int(y2*downscale2)}" for x2,y2 in pts2)

        # ---------- encode the background ------------------------------------------
        st.session_state.img_h2, st.session_state.img_w2 = st.session_state.image_original.shape[:2]
        st.session_state.img_h2 = st.session_state.img_h2 *0.67
        st.session_state.img_w2 = st.session_state.img_w2 * 0.67
        pil_img2  = Image.fromarray(st.session_state.image_combined)          
        buff2     = io.BytesIO()
        pil_img2.save(buff2, format="JPEG")
        st.session_state.img_b642  = base64.b64encode(buff2.getvalue()).decode()



        # ---------- build <polygon> list -------------------------------------------
        st.session_state.poly_html2 = ""

        # (1) matched pairs
        for label_idx2,(nuc_idx2,cyt_idx2,nc_ratio2) in enumerate(test_rst, start=1):
            color2   = get_gradient_color(nc_ratio2)          
            css_col2 = rgb_to_css2(color2)
            fill_color = rgba_str(color2, 0.2)
            stroke_color = rgba_str(color2, 0.5)
            hover_fill = rgba_str(color2, 0.2)
            st.session_state.points_nuc2 = poly_points_str2(pred_masks[nuc_idx2])
            st.session_state.points_cyt2 = poly_points_str2(pred_masks[cyt_idx2])

            st.session_state.poly_html2 += f'''
            <polygon points="{st.session_state.points_cyt2}"
                    class="obj2 hoverable2"
                    data-id2="{label_idx2}"
                    data-ratio2="{nc_ratio2:.3f}"
                    style="fill:{rgba_str(color2, 0.2)}; stroke:{stroke_color}; stroke-width:2; fill-opacity:0.1">
            </polygon>'''

            st.session_state.poly_html2 += f'''
            <polygon points="{st.session_state.points_nuc2}"
                    class="obj2 hoverable2 nucleus2"
                    data-id2="{label_idx2}"
                    data-ratio2="{nc_ratio2:.3f}"
                    style="fill:{rgba_str(color2, 0.2)}; stroke:{stroke_color}; stroke-width:2; fill-opacity:0.1">
            </polygon>'''

        # (2) unmatched nuclei
        for idx2 in unmatched_nuclei:
            ratio2     = 0.0
            color2     = (255 , 0, 0)
            fill_color = rgba_str(color2, 0.2)
            stroke_color = rgba_str(color2, 0.5)
            hover_fill = rgba_str(color2, 0.2)
            css_col2   = rgb_to_css2(color2)
            st.session_state.points_n2  = poly_points_str2(pred_masks[idx2])
            st.session_state.poly_html2 += f'''
            <polygon points="{st.session_state.points_n2}"
                    class="obj2 hoverable2 nucleus2"
                    data-id2="U{idx2}"
                    data-ratio2="Unmatched"
                    style="fill:{rgba_str(color2, 0.2)}; stroke:{stroke_color}; stroke-width:2; fill-opacity:0.1">
            </polygon>'''

        # ---------- compose HTML ----------------------------------------------------
        st.session_state.html_code2 = f"""
        <style>
        .container2 {{
        position:relative;
        width:{st.session_state.img_w2}px;
        height:{st.session_state.img_h2}px;
        font-family:Arial, sans-serif;
        }}
        .hoverable2 {{ transition:fill .15s, stroke .15s; cursor:pointer; }}
        .hoverable2:hover {{ filter: brightness(1.6); }}
        .tooltip2 {{
        position:absolute; background:#000; color:#fff; padding:4px 8px;
        border-radius:4px; font-size:12px; display:none; pointer-events:none;
        white-space:nowrap; z-index:20;
        }}
        .obj2:hover {{
            transition: fill 1s ease, stroke 0.2s ease;
            fill-opacity: 1 !important;
            cursor: pointer;
        }}
        </style>

        <div class="container2">
        <img src="data:image/png;base64,{st.session_state.img_b642}" width="{st.session_state.img_w2}" height="{st.session_state.img_h2}">
        <svg width="{st.session_state.img_w2}" height="{st.session_state.img_h2}" style="position:absolute; top:0; left:0;">
            {st.session_state.poly_html2}
        </svg>
        <div id="tip2" class="tooltip2"></div>
        </div>

        <script>
        const tip2 = document.getElementById("tip2");
        document.querySelector("svg").addEventListener("mousemove", e => {{
            const tgt2 = e.target;
            if(tgt2.tagName === "polygon") {{
                tip2.style.display = "block";
                tip2.style.left = (e.pageX+12)+'px';
                tip2.style.top  = (e.pageY+12)+'px';
                tip2.textContent = `Obj #${{tgt2.dataset.id2}} NC ratio: ${{tgt2.dataset.ratio2}}`;
            }} else {{
                tip2.style.display = "none";
            }}
        }});

        </script>
        """
    st.session_state.count_nucl_min = min(st.session_state.count_nucl)
    st.session_state.count_nucl_max = max(st.session_state.count_nucl)
    st.session_state.count_cyto_min = min(st.session_state.count_cyto)
    st.session_state.count_cyto_max = max(st.session_state.count_cyto)
    st.session_state.nc_ratio_average = np.mean(st.session_state.nc_ratio_)
    # st.session_state.nc_ratio_arr = np.array(st.session_state.nc_ratio_)
    st.session_state.nc_ratio_min = min(st.session_state.nc_ratio_)
    st.session_state.nc_ratio_max = max(st.session_state.nc_ratio_)
    bins = [0.0, 0.3, 0.5, 0.7, 1.0]
    arr = np.array(st.session_state.nc_ratio_)
    st.session_state.counts, _ = np.histogram(arr, bins=bins)
    st.session_state.total_rows = len(st.session_state.matched_pairs)
    st.session_state.gradient_bar_html = """
        <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
            <span style="min-width: 50px; text-align: right;">0.0</span>
            <div style="
                flex-grow: 1;
                height: 20px;
                background: linear-gradient(to right, green, #FFFF00, #FF0000);
                border-radius: 4px;
            "></div>
            <span style="min-width: 50px; text-align: left;">1.0</span>
        </div>
        """






#########################################################################################################################################################################################
    # สร้างสรรค์ Streamlit
if "vis_image" in st.session_state:

    if "selected_image" not in st.session_state:
        st.session_state.selected_image = "original"

    # Option labels and mapping
    option_labels = ["Original", "Mask", "Matched", "Prepare"] #"Unmatched", "Combined", "Prepare"]
    option_map = {
        "Original": "original",
        "Mask": "mask",
        "Matched": "matched",
        # "Unmatched": "unmatched",
        # "Combined": "combined",
        "Prepare": "prepare",
    }

    # Layout: 2 rows, 3 columns
    cols = st.columns(3)
    for i in range(3):
        if cols[i].button(option_labels[i], use_container_width=True):
            st.session_state.selected_image = option_map[option_labels[i]]

    cols = st.columns(1)
    for i in range(3, 4):
        if cols[i - 3].button(option_labels[i], use_container_width=True):
            st.session_state.selected_image = option_map[option_labels[i]]

    if st.session_state.selected_image == "original":
        st.markdown("---")
        st.text("Original")
        st.components.v1.html(st.session_state.html_code, height=st.session_state.image_test.height*0.7)
        # st.image(st.session_state.image_original, caption="Original Image", use_container_width=True)
    elif st.session_state.selected_image == "mask":
        st.markdown("---")
        st.text("Mask")
        st.image(st.session_state.vis_image, use_container_width=True)
    elif st.session_state.selected_image == "matched":
        # st.image(st.session_state.image_display, caption="Visualized with Mask R-CNN", use_container_width=True)
        st.markdown("---")
        st.text("Matched")
        st.components.v1.html(st.session_state.html_code2, height=st.session_state.img_h2)
        # Define gradient with 3 colors
        st.markdown(st.session_state.gradient_bar_html, unsafe_allow_html=True)
    # elif st.session_state.selected_image == "unmatched":
    #     st.image(st.session_state.image_unmatched, use_container_width=True)
    # elif st.session_state.selected_image == "combined":
    #     st.image(st.session_state.image_combined, caption="image_combined", use_container_width=True)
    elif st.session_state.selected_image == "prepare":
        st.markdown("---")
        st.text("Prepare")
        # st.image(st.session_state.image_display, caption="image_display", use_container_width=True)
        
        if "selections2" not in st.session_state:
            st.session_state["selections2"] = [False] * 3

        options2 = ["Original", "Mask", "Matched"]#, "Unmatched", "Combined"]
        option_meanings2 = {
            "Original": st.session_state.image_original,
            "Mask": st.session_state.vis_image,
            "Matched": st.session_state.image_display,
            # "Unmatched": st.session_state.image_unmatched,
            # "Combined": st.session_state.image_combined
        }
        cols2 = st.columns(3)

        # Update selections
        for i2 in range(3):
            with cols2[i2]:
                st.session_state["selections2"][i2] = st.checkbox(options2[i2], value=st.session_state["selections2"][i2], key=f"checkbox2_{i2}")

        # Count selected
        selected_count2 = sum(st.session_state["selections2"])

        # Enforce max 2 selections
        if selected_count2 > 2:
            st.warning("Please select only up to 2 options.")

        selected_options2 = [opt2 for opt2, checked2 in zip(options2, st.session_state["selections2"]) if checked2]
        # st.write("Selected:", selected_options2)
        if len(selected_options2) == 1:
            st.image(option_meanings2[selected_options2[0]])
        elif len(selected_options2) == 2:
            img1 = option_meanings2[selected_options2[0]]
            img2 = option_meanings2[selected_options2[1]]
            label1 = selected_options2[0]
            label2 = selected_options2[1]
            image_comparison(
                img1=img1,
                img2=img2,
                label1=label1,
                label2=label2)

    st.markdown("---")
    st.text("Conclude")
    col1_, col2_ = st.columns(2)
    with col1_:
        st.text("Nuclues : " + str(len(st.session_state.count_nucl)) + "\t Cytoplasm : " + str(len(st.session_state.count_cyto)))
        st.text(f"N/C ratio \n\taverage: {st.session_state.nc_ratio_average:.2f} \n\tmin: {st.session_state.nc_ratio_min:.2f} \n\tmax: {st.session_state.nc_ratio_max:.2f}")
        
        st.text(f"Counts by N/C ratio:\n\t0.0 - 0.3 : {st.session_state.counts[0]}\n\t0.3 - 0.5 : {st.session_state.counts[1]}\n\t0.5 - 0.7 : {st.session_state.counts[2]}\n\t0.7 - 1.0 : {st.session_state.counts[3]}")
    with col2_:
        st.text("Matched pairs : " + str(len(st.session_state.matched_pairs)))
        st.text(f"Nucleus Area \n\taverage: {np.mean(st.session_state.count_nucl):.2f} \n\t"
            f"max: {st.session_state.count_nucl_max:.2f} \n\t"
            f"min: {st.session_state.count_nucl_min:.2f}")
        st.text(f"Cytoplasm Area \n\taverage: {np.mean(st.session_state.count_cyto):.2f} \n\t"
            f"max: {st.session_state.count_cyto_max:.2f} \n\t"
            f"min: {st.session_state.count_cyto_min:.2f}")
    st.markdown("---")
    st.text("\nDetails")
    # col_1, col_2, col_3, col_4 = st.columns(4)
    # with col_1:
    #     st.text("Pair No.")
    #     for i in range(len(st.session_state.matched_pairs)):
    #         st.text(str(i+1)+"\n")
    # with col_2:
    #     st.text("N/C ratio")
    #     for nc_ratio in session_state.nc_ratio_:
    #         st.text(f"{nc_ratio:.2f}")
    # with col_3:
    #     st.text("Nucleus area")
    #     for area_nuc in st.session_state.pairs_nuc:
    #         st.text(f"{area_nuc:.2f}   \tpixels")
    # with col_4:
    #     st.text("Cytoplasm area")
    #     for area_cyt in st.session_state.pairs_cyt:
    #         st.text(f"{area_cyt:.2f} \tpixels")

    if "num_rows_shown" not in st.session_state:
        st.session_state.num_rows_shown = 5
    # Get the number of rows to display
    num_rows = st.session_state.num_rows_shown

    # Slicing the data
    ratios = st.session_state.nc_ratio_[:num_rows]
    nuc_areas = st.session_state.pairs_nuc[:num_rows]
    cyt_areas = st.session_state.pairs_cyt[:num_rows]
    matched = st.session_state.matched_pairs[:num_rows]

    # Layout
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        st.text("Pair No.")
        for i in range(len(matched)):
            st.text(str(i+1))
    with col_2:
        st.text("N/C ratio")
        for nc_ratio in ratios:
            st.text(f"{nc_ratio:.2f}")
    with col_3:
        st.text("Nucleus area")
        for area_nuc in nuc_areas:
            st.text(f"{area_nuc} px")
    with col_4:
        st.text("Cytoplasm area")
        for area_cyt in cyt_areas:
            st.text(f"{area_cyt} px")
        

    # Button to show more

    # Show button only if more rows are available
    if st.session_state.num_rows_shown < st.session_state.total_rows:
        if st.button("Show more"):
            st.session_state.num_rows_shown = min(
                st.session_state.num_rows_shown + 5, st.session_state.total_rows
            )



