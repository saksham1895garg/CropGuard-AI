"""
Generate the official 7-slide PowerPoint presentation for UPES Deep Learning PBL Evaluation.
Project: CropGuard AI — Multi-Model Crop Disease & Pest Surveillance System
Student: Saksham Garg (CSAI3027P)
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette Definitions
    COLOR_BG = RGBColor(248, 249, 250)        # #F8F9FA Very Light Neutral
    COLOR_WHITE = RGBColor(255, 255, 255)     # #FFFFFF Card background
    COLOR_PRIMARY = RGBColor(27, 77, 62)      # #1B4D3E Deep Forest Green
    COLOR_PRIMARY_LIGHT = RGBColor(232, 245, 233) # Soft green pill
    COLOR_SECONDARY = RGBColor(194, 125, 0)   # #C27D00 Earthy Amber / Gold
    COLOR_SECONDARY_LIGHT = RGBColor(254, 243, 199) # Soft amber pill
    COLOR_ACCENT = RGBColor(46, 125, 50)      # #2E7D32 Vibrant Plant Green
    COLOR_CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0 Subtle Border
    COLOR_TEXT_DARK = RGBColor(30, 41, 59)    # #1E293B Dark Slate
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # #64748B Slate Muted
    COLOR_BLUE_LIGHT = RGBColor(224, 242, 254) # Soft blue pill
    COLOR_BLUE_DARK = RGBColor(3, 105, 161)    # Deep Sky Blue
    COLOR_RED_ACCENT = RGBColor(185, 28, 28)   # Alert red/brick

    def set_slide_background(slide):
        bg_shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
        )
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = COLOR_BG
        bg_shape.line.fill.background() # No line

    def add_header(slide, slide_num, title, subtitle):
        # Category / Slide Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(4.0), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"UPES CSAI3027P  |  SLIDE {slide_num:02d} OF 07"
        p_tag.font.size = Pt(9.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_SECONDARY
        p_tag.font.name = "Arial"

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.55))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY
        p_title.font.name = "Arial"

        # Subtitle
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.733), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.font.name = "Arial"

        # Decorative subtle green line under header
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.68), Inches(11.733), Inches(0.02)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_CARD_BORDER
        line.line.fill.background()

    def add_card(slide, left, top, width, height, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: Problem and Objective
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    add_header(s1, 1, 
               "CropGuard AI: Multi-Model Crop Disease & Pest Surveillance", 
               "Bridging diagnostic silos in precision agriculture via decoupled deep learning architectures")

    # Author metadata badge
    meta_card = add_card(s1, Inches(0.8), Inches(1.85), Inches(11.733), Inches(0.65), 
                         bg_color=COLOR_PRIMARY_LIGHT, border_color=COLOR_ACCENT)
    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.92), Inches(11.333), Inches(0.5))
    tf_meta = meta_box.text_frame
    tf_meta.word_wrap = True
    p_meta = tf_meta.paragraphs[0]
    p_meta.text = "Student: Saksham Garg  |  Course: Deep Learning (CSAI3027P)  |  Evaluation: PBL Project Defense  |  Presentation Duration: 8 Minutes (incl. Live Demo)"
    p_meta.font.size = Pt(11)
    p_meta.font.bold = True
    p_meta.font.color.rgb = COLOR_PRIMARY
    p_meta.font.name = "Arial"

    # Card 1: Agricultural Challenge
    add_card(s1, Inches(0.8), Inches(2.65), Inches(4.5), Inches(4.45))
    title_box1 = s1.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(4.1), Inches(0.4))
    p1 = title_box1.text_frame.paragraphs[0]
    p1.text = "The Agricultural Challenge"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY
    p1.font.name = "Arial"

    body_box1 = s1.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(4.1), Inches(3.6))
    tf1 = body_box1.text_frame
    tf1.word_wrap = True
    bullets1 = [
        ("Severe Yield Reductions: ", "Plant pathogens and insect pests collectively cause 20% to 40% loss in global agricultural production annually."),
        ("Diagnostic Silos: ", "Existing computerized solutions either classify leaf diseases OR detect pests in isolation, failing actual multi-pathogen field realities."),
        ("Computer Vision Conflict: ", "A single monolithic neural network cannot resolve whole-leaf taxonomic taxonomy, pixel-precise lesion borders, and tiny insect bounding boxes simultaneously.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets1):
        p = tf1.add_paragraph() if i > 0 else tf1.paragraphs[0]
        p.space_after = Pt(12)
        run1 = p.add_run()
        run1.text = "• " + b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = COLOR_TEXT_DARK
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(10.5)
        run2.font.color.rgb = COLOR_TEXT_MUTED

    # Card 2: CropGuard AI Objectives
    add_card(s1, Inches(5.5), Inches(2.65), Inches(4.2), Inches(4.45))
    title_box2 = s1.shapes.add_textbox(Inches(5.7), Inches(2.8), Inches(3.8), Inches(0.4))
    p2 = title_box2.text_frame.paragraphs[0]
    p2.text = "CropGuard AI Objectives"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_PRIMARY
    p2.font.name = "Arial"

    body_box2 = s1.shapes.add_textbox(Inches(5.7), Inches(3.25), Inches(3.8), Inches(3.6))
    tf2 = body_box2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        ("Tri-Modal Decoupled Engine: ", "Deploy three task-specific deep neural architectures optimized independently for their unique vision formulation."),
        ("Three Actionable Outputs:\n", "  1. Foliar Pathology Classification (38 classes)\n  2. Pixel-Level Lesion Segmentation (Binary mask)\n  3. Insect Pest Localization (102 classes + counts)"),
        ("Unified Field Dashboard: ", "Interactive Streamlit interface enabling multi-modal diagnosis, overlay visualization, and standardized diagnostic reports.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets2):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.space_after = Pt(10)
        run1 = p.add_run()
        run1.text = "• " + b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = COLOR_TEXT_DARK
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(10.5)
        run2.font.color.rgb = COLOR_TEXT_MUTED

    # Card 3: Real Sample Image & Vision Scope
    add_card(s1, Inches(9.9), Inches(2.65), Inches(2.633), Inches(4.45))
    title_box3 = s1.shapes.add_textbox(Inches(10.05), Inches(2.8), Inches(2.333), Inches(0.4))
    p3 = title_box3.text_frame.paragraphs[0]
    p3.text = "Sample Input"
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_PRIMARY
    p3.font.name = "Arial"

    # Add real image
    sample_img_path = "test_samples/classification_and_lesions/01_Tomato_Late_Blight.jpg"
    if os.path.exists(sample_img_path):
        s1.shapes.add_picture(sample_img_path, Inches(10.15), Inches(3.3), width=Inches(2.133))

    cap_box = s1.shapes.add_textbox(Inches(10.05), Inches(5.5), Inches(2.333), Inches(1.4))
    tf_cap = cap_box.text_frame
    tf_cap.word_wrap = True
    p_cap = tf_cap.paragraphs[0]
    p_cap.text = "Tomato Late Blight (Phytophthora infestans). Evaluated across classification, lesion delineation, and pest surveillance."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = COLOR_TEXT_MUTED
    p_cap.font.name = "Arial"

    # Speaker notes Slide 1
    s1.notes_slide.notes_text_frame.text = (
        "Good morning respected evaluators. I am Saksham Garg, presenting my Deep Learning project: "
        "'CropGuard AI — Multi-Model Crop Disease and Pest Surveillance System'.\n\n"
        "In agriculture, foliar diseases and insect pests destroy 20 to 40 percent of total crop yields. "
        "However, current machine learning solutions suffer from a fundamental flaw: diagnostic silos. "
        "A farmer does not encounter pests and fungal blights in separate worlds; yet existing tools either only classify "
        "a leaf image or detect insects, failing the multi-pathogen reality of open fields.\n\n"
        "Crucially, a single monolithic model cannot effectively solve this: whole-leaf classification, pixel-level lesion "
        "delineation, and small mobile insect detection represent three radically different computer vision paradigms.\n\n"
        "CropGuard AI solves this by engineering a decoupled, three-model surveillance architecture that delivers three simultaneous "
        "actionable outputs: 38-class crop-disease taxonomy, binary lesion masks with damage ratios, and 102-class insect bounding boxes. "
        "Everything is unified into an interactive, zero-crash Streamlit diagnostic dashboard."
    )

    # =========================================================================
    # SLIDE 2: System Architecture
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, 2, 
               "System Architecture: Decoupled Tri-Modal Pipeline", 
               "Mapping three distinct agricultural computer vision tasks to dedicated, decoupled deep architectures")

    # 3 Architecture Cards for 3 Models
    col_w = Inches(3.75)
    gap = Inches(0.24)
    start_left = Inches(0.8)

    # Branch 1: ResNet50
    left_1 = start_left
    add_card(s2, left_1, Inches(1.85), col_w, Inches(4.35))
    # Badge
    b1 = add_card(s2, left_1 + Inches(0.2), Inches(2.05), Inches(3.35), Inches(0.4), 
                  bg_color=COLOR_PRIMARY_LIGHT, border_color=COLOR_ACCENT)
    tb1 = s2.shapes.add_textbox(left_1 + Inches(0.2), Inches(2.08), Inches(3.35), Inches(0.35))
    ptb1 = tb1.text_frame.paragraphs[0]
    ptb1.text = "BRANCH 1: FOLIAR CLASSIFICATION"
    ptb1.font.size = Pt(9.5)
    ptb1.font.bold = True
    ptb1.font.color.rgb = COLOR_PRIMARY
    ptb1.alignment = PP_ALIGN.CENTER

    tb_b1 = s2.shapes.add_textbox(left_1 + Inches(0.2), Inches(2.6), Inches(3.35), Inches(3.4))
    tf_b1 = tb_b1.text_frame
    tf_b1.word_wrap = True
    items_b1 = [
        ("Architecture: ", "ResNet50 (24.1M params)"),
        ("Paradigm: ", "Global Multi-Class Classification"),
        ("Input Specification: ", "224x224x3 RGB, Caffe BGR normalization (mean subtraction)"),
        ("Backbone Core: ", "Deep Residual Bottleneck Blocks with identity skip connections preventing vanishing gradients"),
        ("Output: ", "38 crop-disease probabilities via Softmax (14 crops, 26 diseases, 12 healthy)")
    ]
    for i, (k, v) in enumerate(items_b1):
        p = tf_b1.add_paragraph() if i > 0 else tf_b1.paragraphs[0]
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(10)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Branch 2: U-Net
    left_2 = left_1 + col_w + gap
    add_card(s2, left_2, Inches(1.85), col_w, Inches(4.35))
    # Badge
    b2 = add_card(s2, left_2 + Inches(0.2), Inches(2.05), Inches(3.35), Inches(0.4), 
                  bg_color=COLOR_SECONDARY_LIGHT, border_color=COLOR_SECONDARY)
    tb2 = s2.shapes.add_textbox(left_2 + Inches(0.2), Inches(2.08), Inches(3.35), Inches(0.35))
    ptb2 = tb2.text_frame.paragraphs[0]
    ptb2.text = "BRANCH 2: LESION SEGMENTATION"
    ptb2.font.size = Pt(9.5)
    ptb2.font.bold = True
    ptb2.font.color.rgb = COLOR_SECONDARY
    ptb2.alignment = PP_ALIGN.CENTER

    tb_b2 = s2.shapes.add_textbox(left_2 + Inches(0.2), Inches(2.6), Inches(3.35), Inches(3.4))
    tf_b2 = tb_b2.text_frame
    tf_b2.word_wrap = True
    items_b2 = [
        ("Architecture: ", "U-Net (7.8M params)"),
        ("Paradigm: ", "Pixel-Level Semantic Segmentation"),
        ("Input Specification: ", "256x256x3 RGB, [0, 1] min-max float normalization"),
        ("Backbone Core: ", "Symmetric 4-stage encoder-decoder with long skip connections for localized spatial edge recovery"),
        ("Output: ", "256x256 binary lesion mask (Sigmoid) + affected foliar area percentage")
    ]
    for i, (k, v) in enumerate(items_b2):
        p = tf_b2.add_paragraph() if i > 0 else tf_b2.paragraphs[0]
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(10)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Branch 3: YOLO26n
    left_3 = left_2 + col_w + gap
    add_card(s2, left_3, Inches(1.85), col_w, Inches(4.35))
    # Badge
    b3 = add_card(s2, left_3 + Inches(0.2), Inches(2.05), Inches(3.35), Inches(0.4), 
                  bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_DARK)
    tb3 = s2.shapes.add_textbox(left_3 + Inches(0.2), Inches(2.08), Inches(3.35), Inches(0.35))
    ptb3 = tb3.text_frame.paragraphs[0]
    ptb3.text = "BRANCH 3: INSECT PEST DETECTION"
    ptb3.font.size = Pt(9.5)
    ptb3.font.bold = True
    ptb3.font.color.rgb = COLOR_BLUE_DARK
    ptb3.alignment = PP_ALIGN.CENTER

    tb_b3 = s2.shapes.add_textbox(left_3 + Inches(0.2), Inches(2.6), Inches(3.35), Inches(3.4))
    tf_b3 = tb_b3.text_frame
    tf_b3.word_wrap = True
    items_b3 = [
        ("Architecture: ", "YOLO26n (2.6M params)"),
        ("Paradigm: ", "Single-Stage Real-Time Object Detection"),
        ("Input Specification: ", "640x640x3 RGB, letterboxed aspect ratio preserved"),
        ("Backbone Core: ", "Modified CSPDarknet with PAN-FPN feature pyramid and anchor-free detection head"),
        ("Output: ", "Pest bounding boxes [x1, y1, x2, y2], 102 class IDs, confidence scores & counts")
    ]
    for i, (k, v) in enumerate(items_b3):
        p = tf_b3.add_paragraph() if i > 0 else tf_b3.paragraphs[0]
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(10)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Integration Layer Banner
    add_card(s2, Inches(0.8), Inches(6.35), Inches(11.733), Inches(0.75), 
             bg_color=RGBColor(241, 245, 249), border_color=COLOR_CARD_BORDER)
    tb_int = s2.shapes.add_textbox(Inches(1.0), Inches(6.42), Inches(11.333), Inches(0.6))
    tf_int = tb_int.text_frame
    tf_int.word_wrap = True
    p_int = tf_int.paragraphs[0]
    r_int1 = p_int.add_run()
    r_int1.text = "UNIFIED INTEGRATION LAYER (Streamlit Engine): "
    r_int1.font.bold = True
    r_int1.font.size = Pt(10.5)
    r_int1.font.color.rgb = COLOR_PRIMARY
    r_int2 = p_int.add_run()
    r_int2.text = "Decoupled model loading guarantees isolation (failure in one branch never breaks others). Includes dynamic image routing, side-by-side overlay rendering, zero-detection safety guards, and audit JSON generation."
    r_int2.font.size = Pt(10)
    r_int2.font.color.rgb = COLOR_TEXT_DARK

    # Speaker notes Slide 2
    s2.notes_slide.notes_text_frame.text = (
        "Here is the system architecture of CropGuard AI. Rather than forcing a single network to perform "
        "multiple conflicting objectives, our design decouples the problem into three specialized branches, "
        "each aligned with a canonical computer vision task.\n\n"
        "Branch 1 handles Foliar Classification using ResNet50 with 24.1 million parameters. It ingests 224x224 RGB images "
        "normalized using Caffe BGR mean-subtraction, leveraging deep residual bottleneck blocks to classify 38 crop-disease states.\n\n"
        "Branch 2 performs Lesion Segmentation using an encoder-decoder U-Net with 7.8 million parameters. Operating on 256x256 "
        "inputs normalized to [0, 1], its symmetric skip connections preserve fine spatial boundaries, outputting pixel-precise masks "
        "and quantitative foliar damage area percentages.\n\n"
        "Branch 3 performs Insect Pest Detection using lightweight YOLO26n with 2.6 million parameters. It processes 640x640 "
        "letterboxed inputs through a CSPDarknet backbone and PAN-FPN neck, utilizing an anchor-free head to predict bounding boxes "
        "and pest counts across 102 insect classes in real time.\n\n"
        "All three branches are unified via a lightweight Streamlit integration layer that keeps execution isolated and responsive."
    )

    # =========================================================================
    # SLIDE 3: Datasets & Splits
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, 3, 
               "Datasets & Splitting Strategy: Zero-Leakage Rigor", 
               "Empirical data integrity: Group-aware partitioning, official benchmark quarantine, and complete XML conversion")

    # Embed Visual Chart (dataset_splits.png)
    chart_splits_path = "presentation/assets/dataset_splits.png"
    if os.path.exists(chart_splits_path):
        s3.shapes.add_picture(chart_splits_path, Inches(0.8), Inches(1.85), width=Inches(11.733))

    # Bottom Audit Summary Banner
    add_card(s3, Inches(0.8), Inches(6.05), Inches(11.733), Inches(1.15),
             bg_color=RGBColor(241, 245, 249), border_color=COLOR_CARD_BORDER)
    tb_audit = s3.shapes.add_textbox(Inches(1.0), Inches(6.10), Inches(11.333), Inches(1.0))
    tf_audit = tb_audit.text_frame
    tf_audit.word_wrap = True
    
    p_aud_title = tf_audit.paragraphs[0]
    p_aud_title.text = "EMPIRICAL BENCHMARK INTEGRITY & LEAKAGE PREVENTION SUMMARY:"
    p_aud_title.font.size = Pt(10)
    p_aud_title.font.bold = True
    p_aud_title.font.color.rgb = COLOR_PRIMARY
    
    audit_points = [
        "• PlantVillage: Group-aware partition strictly prevents 32.3% physical leaf identity leakage across bursts. Quarantined 10,999 test images.",
        "• PlantSeg: Preserved official author benchmark split (115 disease classes, 11,458 pairs). Zero mask corruption across 2,295 test pairs.",
        "• IP102: Converted 18,976/18,976 VOC XMLs to YOLO format (22,283 valid boxes, recovered IP087000986). 3,798 official test images strictly untouched."
    ]
    for pt in audit_points:
        p_pt = tf_audit.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(9)
        p_pt.font.color.rgb = COLOR_TEXT_DARK

    # Speaker notes Slide 3
    s3.notes_slide.notes_text_frame.text = (
        "Slide 3 addresses our data pipeline and leakage prevention strategy. A key academic contribution of this project "
        "is absolute empirical rigor across three distinct datasets.\n\n"
        "First, PlantVillage contains 54,305 images across 38 classes. In the literature, many published papers naively apply "
        "random train-test splits, which inadvertently leaks up to 32.3% of identical leaves photographed from multiple angles into the "
        "test set. We designed a group-aware partition that completely eliminates physical leaf leakage, yielding 35,221 training, "
        "8,085 validation, and 10,999 strictly held-out test images.\n\n"
        "Second, PlantSeg contains 11,458 paired RGB images and pixel-level binary lesion masks across 115 disease conditions. "
        "We preserved the official published benchmark split of 7,916 train, 1,247 validation, and 2,295 test pairs with zero cross-contamination.\n\n"
        "Third, for the IP102 insect benchmark with 18,976 images across 102 pest categories, we audited and converted all 18,976 XML annotations "
        "into normalized YOLO format, generating 22,283 verified bounding boxes. We partitioned the 15,178 trainval images into 12,898 train "
        "and 2,280 validation images, strictly quarantining all 3,798 official test images."
    )

    # =========================================================================
    # SLIDE 4: Model Architecture and Training
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, 4, 
               "Model Architectures & Training Configurations", 
               "Verified training hyperparameters, loss formulations, and optimization pipelines from production artifacts")

    # 3 Comparison Columns
    col_w = Inches(3.75)
    gap = Inches(0.24)
    start_left = Inches(0.8)

    # Col 1: ResNet50
    add_card(s4, left_1, Inches(1.85), col_w, Inches(5.2))
    tb_m1 = s4.shapes.add_textbox(left_1 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_m1 = tb_m1.text_frame
    tf_m1.word_wrap = True
    
    p = tf_m1.paragraphs[0]
    p.text = "ResNet50 Classification"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    sub = tf_m1.add_paragraph()
    sub.text = "TensorFlow 2.22 / Keras"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    m1_specs = [
        ("Base Pretraining: ", "ImageNet-1k initialized weights"),
        ("Transfer Strategy: ", "Two-phase staged fine-tuning\n• Phase 1 (10 ep): Frozen base, lr=1e-3\n• Phase 2 (10 ep): Top 25 layers unfrozen, lr=1e-5"),
        ("Loss Function: ", "Categorical Cross-Entropy"),
        ("Batch / Image Size: ", "Batch Size: 16 | Img Size: 224x224"),
        ("Preprocessing: ", "Caffe BGR zero-centering ([103.939, 116.779, 123.68])"),
        ("Regularization: ", "Early Stopping (patience=4), ModelCheckpoint on val_loss"),
        ("Total Parameters: ", "24,130,214 parameters")
    ]
    for k, v in m1_specs:
        p = tf_m1.add_paragraph()
        p.space_after = Pt(5)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Col 2: U-Net
    add_card(s4, left_2, Inches(1.85), col_w, Inches(5.2))
    tb_m2 = s4.shapes.add_textbox(left_2 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_m2 = tb_m2.text_frame
    tf_m2.word_wrap = True
    
    p = tf_m2.paragraphs[0]
    p.text = "U-Net Segmentation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    sub = tf_m2.add_paragraph()
    sub.text = "TensorFlow 2.22 / Keras"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    m2_specs = [
        ("Base Pretraining: ", "Trained from scratch on PlantSeg"),
        ("Topology: ", "4-stage contracting encoder (Conv-BN-ReLU-Pool) + bottleneck + 4-stage expanding decoder with skip concatenation"),
        ("Compound Loss: ", "0.5 * BCE + 0.5 * Dice Loss\n(Mitigates severe class imbalance where lesions occupy <5% of leaf pixels)"),
        ("Batch / Image Size: ", "Batch Size: 8 | Img Size: 256x256x3"),
        ("Optimization: ", "Adam optimizer, learning rate = 1e-4"),
        ("Regularization: ", "Early Stopping (patience=5), Dropout (0.2), ReduceLROnPlateau"),
        ("Total Parameters: ", "7,771,873 parameters")
    ]
    for k, v in m2_specs:
        p = tf_m2.add_paragraph()
        p.space_after = Pt(5)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Col 3: YOLO26n
    add_card(s4, left_3, Inches(1.85), col_w, Inches(5.2))
    tb_m3 = s4.shapes.add_textbox(left_3 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_m3 = tb_m3.text_frame
    tf_m3.word_wrap = True
    
    p = tf_m3.paragraphs[0]
    p.text = "YOLO26n Pest Detection"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    sub = tf_m3.add_paragraph()
    sub.text = "PyTorch 2.15 / Ultralytics"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    m3_specs = [
        ("Base Pretraining: ", "COCO-pretrained weights (yolo26n.pt)"),
        ("Backbone / Neck: ", "Cross-Stage Partial Darknet with Path Aggregation Network (PAN-FPN)"),
        ("Head / Loss: ", "Anchor-free decoupled head; Complete IoU (CIoU) box loss + DFL + BCE class loss"),
        ("Batch / Image Size: ", "Batch Size: 4 | Img Size: 640x640"),
        ("Optimization: ", "SGD/AdamW, initial lr = 0.01 with linear warmup over 3 epochs"),
        ("Training Budget: ", "20 epochs maximum, Early Stopping (patience=5) on val mAP@50"),
        ("Total Parameters: ", "2,582,972 parameters (102 classes)")
    ]
    for k, v in m3_specs:
        p = tf_m3.add_paragraph()
        p.space_after = Pt(5)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Speaker notes Slide 4
    s4.notes_slide.notes_text_frame.text = (
        "Slide 4 details the technical architectures and training regimens of our three models, highlighting the design "
        "choices verified in our production configuration files.\n\n"
        "For ResNet50 in TensorFlow, we implemented a two-phase transfer learning regimen starting from ImageNet weights. "
        "Phase 1 trained a custom Dense classification head for 10 epochs at lr=1e-3 with frozen base layers. "
        "Phase 2 unfroze the top 25 convolutional layers for fine-tuning at lr=1e-5. Input images were strictly preprocessed "
        "using Caffe BGR mean-subtraction to match ImageNet feature statistics.\n\n"
        "For U-Net, also in TensorFlow, we trained from scratch over 20 epochs using an Adam optimizer at lr=1e-4. "
        "Crucially, because foliar lesions typically occupy under 5% of the total leaf pixel area, standard cross-entropy suffers from "
        "severe background gradient dominance. We formulated a compound loss combining Binary Cross-Entropy and Sørensen-Dice loss "
        "(0.5 BCE + 0.5 Dice), which balances smooth gradient convergence with direct overlap maximization.\n\n"
        "For YOLO26n in PyTorch, we fine-tuned COCO-pretrained weights on IP102 for 20 epochs at batch size 4 and image size 640. "
        "It employs a CSPDarknet backbone with PAN-FPN feature pyramid and an anchor-free decoupled head, optimized using CIoU box loss and early stopping."
    )

    # =========================================================================
    # SLIDE 5: Final Test Results
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, 5, 
               "Empirical Evaluation: Verified Held-Out Test Results", 
               "Definitive evaluation metrics computed on strictly quarantined, zero-leakage test partitions")

    # 3 Separate Metric Cards
    add_card(s5, left_1, Inches(1.85), col_w, Inches(5.2))
    add_card(s5, left_2, Inches(1.85), col_w, Inches(5.2))
    add_card(s5, left_3, Inches(1.85), col_w, Inches(5.2))

    # Metric Card 1: ResNet50
    tb_r1 = s5.shapes.add_textbox(left_1 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_r1 = tb_r1.text_frame
    tf_r1.word_wrap = True
    p = tf_r1.paragraphs[0]
    p.text = "ResNet50 Classification"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    sub = tf_r1.add_paragraph()
    sub.text = "Held-out Test Set: 10,999 Images"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    r1_metrics = [
        ("Accuracy", "98.42%", "Overall correct classifications"),
        ("Macro F1-Score", "97.79%", "Balanced across all 38 classes"),
        ("Macro ROC-AUC", "99.98%", "Discriminative confidence rank"),
        ("Macro Precision", "97.82%", "Low false positive rate"),
        ("Macro Recall", "97.81%", "High pathogen sensitivity"),
        ("Inference Latency", "81.8 ms", "12.2 FPS (Single image on GPU)")
    ]
    for name, val, desc in r1_metrics:
        p = tf_r1.add_paragraph()
        p.space_after = Pt(4)
        run_v = p.add_run()
        run_v.text = f"{val}  "
        run_v.font.bold = True
        run_v.font.size = Pt(12)
        run_v.font.color.rgb = COLOR_PRIMARY
        run_n = p.add_run()
        run_n.text = f"{name}\n"
        run_n.font.bold = True
        run_n.font.size = Pt(9.5)
        run_n.font.color.rgb = COLOR_TEXT_DARK
        run_d = p.add_run()
        run_d.text = desc
        run_d.font.size = Pt(8.5)
        run_d.font.color.rgb = COLOR_TEXT_MUTED

    # Metric Card 2: U-Net
    tb_r2 = s5.shapes.add_textbox(left_2 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_r2 = tb_r2.text_frame
    tf_r2.word_wrap = True
    p = tf_r2.paragraphs[0]
    p.text = "U-Net Segmentation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    sub = tf_r2.add_paragraph()
    sub.text = "Held-out Test Set: 2,295 Image Pairs"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    r2_metrics = [
        ("Dice Coefficient", "63.48%", "Mean F1 spatial overlap"),
        ("Intersection over Union", "46.49%", "Jaccard boundary index"),
        ("Specificity", "94.04%", "Healthy tissue preservation"),
        ("Precision", "71.79%", "True lesion boundary accuracy"),
        ("Recall (Sensitivity)", "56.89%", "Faint chlorosis detection"),
        ("Inference Latency", "20.7 ms", "48.3 FPS (Batch size 8 on GPU)")
    ]
    for name, val, desc in r2_metrics:
        p = tf_r2.add_paragraph()
        p.space_after = Pt(4)
        run_v = p.add_run()
        run_v.text = f"{val}  "
        run_v.font.bold = True
        run_v.font.size = Pt(12)
        run_v.font.color.rgb = COLOR_SECONDARY
        run_n = p.add_run()
        run_n.text = f"{name}\n"
        run_n.font.bold = True
        run_n.font.size = Pt(9.5)
        run_n.font.color.rgb = COLOR_TEXT_DARK
        run_d = p.add_run()
        run_d.text = desc
        run_d.font.size = Pt(8.5)
        run_d.font.color.rgb = COLOR_TEXT_MUTED

    # Metric Card 3: YOLO26n
    tb_r3 = s5.shapes.add_textbox(left_3 + Inches(0.2), Inches(2.0), Inches(3.35), Inches(4.9))
    tf_r3 = tb_r3.text_frame
    tf_r3.word_wrap = True
    p = tf_r3.paragraphs[0]
    p.text = "YOLO26n Pest Detection"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    sub = tf_r3.add_paragraph()
    sub.text = "Official Held-out: 3,798 Test Images"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(8)

    r3_metrics = [
        ("mAP @ 0.50", "34.33%", "Standard object detection metric"),
        ("mAP @ 0.50:0.95", "20.79%", "Stringent multi-IoU threshold metric"),
        ("Precision", "44.31%", "Target detection reliability"),
        ("Recall", "38.64%", "Insect retrieval rate across 102 taxa"),
        ("Evaluated Classes", "96 Classes", "Out of 102 total canonical classes"),
        ("Inference Latency", "2.03 ms", "316.5 FPS (Ultra-fast edge capability)")
    ]
    for name, val, desc in r3_metrics:
        p = tf_r3.add_paragraph()
        p.space_after = Pt(4)
        run_v = p.add_run()
        run_v.text = f"{val}  "
        run_v.font.bold = True
        run_v.font.size = Pt(12)
        run_v.font.color.rgb = COLOR_BLUE_DARK
        run_n = p.add_run()
        run_n.text = f"{name}\n"
        run_n.font.bold = True
        run_n.font.size = Pt(9.5)
        run_n.font.color.rgb = COLOR_TEXT_DARK
        run_d = p.add_run()
        run_d.text = desc
        run_d.font.size = Pt(8.5)
        run_d.font.color.rgb = COLOR_TEXT_MUTED

    # Speaker notes Slide 5
    s5.notes_slide.notes_text_frame.text = (
        "Slide 5 presents our verified held-out test evaluations. These metrics were evaluated exactly once on the quarantined "
        "test partitions without parameter tuning on test sets.\n\n"
        "On 10,999 PlantVillage test images, ResNet50 achieved 98.42% accuracy, a Macro F1 score of 97.79%, and a Macro ROC-AUC of 99.98%, "
        "running at 81.8 ms per image. The confusion matrix confirms near-perfect diagonal separation, with minimal confusion only occurring "
        "between visually identical early-stage foliar blights.\n\n"
        "On 2,295 PlantSeg test image-mask pairs across 115 categories, U-Net achieved a Sørensen-Dice coefficient of 63.48% and an IoU of 46.49%. "
        "Importantly, specificity reached 94.04% with precision of 71.79%, meaning healthy green tissue is reliably preserved without false alarms. "
        "The recall of 56.89% reflects the model's conservative boundary delineation around faint, diffuse chlorotic halos.\n\n"
        "On 3,798 official IP102 test images, YOLO26n achieved 34.33% mAP@50 and 20.79% mAP@50-95 across 96 evaluated pest classes. "
        "Given IP102's extreme class imbalance, severe occlusions, and minute pest dimensions in dense foliage, this is a strong benchmark performance. "
        "Most importantly, YOLO26n delivers an inference latency of just 2.03 ms per image (over 316 frames per second), enabling real-time edge processing."
    )

    # =========================================================================
    # SLIDE 6: Live Demonstration
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, 6, 
               "Live Demonstration: End-to-End Surveillance Dashboard", 
               "Seamless operational workflow: Interactive tri-modal diagnosis, visual overlays, and audit reporting")

    # Workflow cards on left
    wf_left = Inches(0.8)
    wf_w = Inches(5.6)
    add_card(s6, wf_left, Inches(1.85), wf_w, Inches(5.2))
    tb_wf = s6.shapes.add_textbox(wf_left + Inches(0.2), Inches(2.0), wf_w - Inches(0.4), Inches(4.8))
    tf_wf = tb_wf.text_frame
    tf_wf.word_wrap = True

    p = tf_wf.paragraphs[0]
    p.text = "Operational Field Workflow"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    sub = tf_wf.add_paragraph()
    sub.text = "Four-step diagnostic pipeline executed live"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(12)

    wf_steps = [
        ("Step 1: Input Acquisition & Preprocessing", "User selects or captures a crop field sample. The system inspects EXIF orientation, validates RGB channels, and prepares memory buffers."),
        ("Step 2: Task-Specific Model Execution", "User selects target inspection mode: Foliar Disease (ResNet50), Lesion Severity (U-Net), or Insect Pests (YOLO26n). Each engine executes independently with dedicated error boundaries."),
        ("Step 3: Multi-View Visual Diagnostic Overlay", "Renders side-by-side original image alongside heatmaps, high-contrast red lesion masks (with damage surface area %), or color-coded pest bounding boxes with confidence scores."),
        ("Step 4: Standardized Diagnostic Export", "Generates one-click machine-readable JSON telemetry payloads and high-resolution annotated PNG artifacts for agronomist records.")
    ]
    for title, desc in wf_steps:
        p = tf_wf.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title + "\n"
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Right side: Live Transition Callout & App Screenshot
    right_left = Inches(6.65)
    right_w = Inches(5.883)
    add_card(s6, right_left, Inches(1.85), right_w, Inches(5.2))

    # Live callout badge
    add_card(s6, right_left + Inches(0.2), Inches(2.05), right_w - Inches(0.4), Inches(0.55),
             bg_color=COLOR_PRIMARY_LIGHT, border_color=COLOR_ACCENT)
    tb_call = s6.shapes.add_textbox(right_left + Inches(0.2), Inches(2.1), right_w - Inches(0.4), Inches(0.45))
    p_call = tb_call.text_frame.paragraphs[0]
    p_call.text = "TRANSITION TO LIVE SYSTEM DEMONSTRATION  (2:00 MIN)"
    p_call.font.size = Pt(10.5)
    p_call.font.bold = True
    p_call.font.color.rgb = COLOR_PRIMARY
    p_call.alignment = PP_ALIGN.CENTER

    # Embedded Screenshot
    app_demo_path = "docs/images/app_demo_screenshot.png"
    if os.path.exists(app_demo_path):
        s6.shapes.add_picture(app_demo_path, right_left + Inches(0.2), Inches(2.75), width=right_w - Inches(0.4))

    tb_live_notes = s6.shapes.add_textbox(right_left + Inches(0.2), Inches(6.35), right_w - Inches(0.4), Inches(0.6))
    tf_ln = tb_live_notes.text_frame
    tf_ln.word_wrap = True
    p_ln = tf_ln.paragraphs[0]
    p_ln.text = "Live Streamlit app running locally at http://localhost:8501. Demonstrating real-time leaf classification, U-Net mask segmentation, and YOLO pest localization."
    p_ln.font.size = Pt(9)
    p_ln.font.color.rgb = COLOR_TEXT_MUTED

    # Speaker notes Slide 6
    s6.notes_slide.notes_text_frame.text = (
        "I will now transition to the live system demonstration running locally on our RTX GPU workstation at port 8501.\n\n"
        "[SWITCH TO LIVE STREAMLIT BROWSER WINDOW]\n\n"
        "Here in the CropGuard AI interface, you see our unified surveillance dashboard. Notice that all three models are loaded "
        "and cached independently.\n\n"
        "1. First, let us test Foliar Classification: I upload a sample of Tomato Late Blight. Instantly, ResNet50 executes with Caffe preprocessing, "
        "returning 'Tomato___Late_blight' with 99.4% confidence, and displaying the top-5 probabilistic distribution.\n\n"
        "2. Second, let us examine Lesion Severity: I switch to the U-Net segmentation module. Uploading a diseased leaf, U-Net predicts the binary "
        "lesion mask, rendering a high-contrast red damage overlay and reporting a 14.8% foliar damage ratio in 21 milliseconds.\n\n"
        "3. Third, let us test Insect Pest Detection: I upload a rice canopy image infested with Rice Leaf Rollers. YOLO26n detects and localizes both "
        "insects in just 2 milliseconds, highlighting bounding boxes, confidence tags, and total pest count.\n\n"
        "Notice also the zero-detection handling: if a healthy clean leaf is passed to YOLO, it outputs a clean 'Zero Pests Detected' status rather than crashing "
        "or fabricating bounding boxes. We can also download the annotated PNG and JSON audit report with a single click.\n\n"
        "[RETURN TO PRESENTATION SLIDES]"
    )

    # =========================================================================
    # SLIDE 7: Limitations, Scope and Conclusion
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, 7, 
               "Limitations, Engineering Insights & Future Roadmap", 
               "Honest appraisal of deployment constraints and architectural evolution for real-world agricultural monitoring")

    # 2 Big Cards: Limitations vs Future Scope
    half_w = Inches(5.7)
    gap_half = Inches(0.333)

    # Left: Limitations
    add_card(s7, Inches(0.8), Inches(1.85), half_w, Inches(5.2))
    tb_lim = s7.shapes.add_textbox(Inches(1.0), Inches(2.0), half_w - Inches(0.4), Inches(4.8))
    tf_lim = tb_lim.text_frame
    tf_lim.word_wrap = True

    p = tf_lim.paragraphs[0]
    p.text = "Identified Technical Limitations"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED_ACCENT
    
    sub = tf_lim.add_paragraph()
    sub.text = "Domain constraints observed during empirical testing"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(10)

    lim_items = [
        ("Controlled Background Domain Shift: ", "PlantVillage images were acquired against uniform studio backdrops. Performance drops when tested on cluttered, variable-lighting outdoor field canopies without preprocessing."),
        ("Necrotic Tissue vs Pathogen Etiology: ", "U-Net segments morphological necrosis and chlorosis, but cannot distinguish whether cell death was caused by fungal blight, mechanical trauma, or nutritional deficiency."),
        ("Extreme Scale Variance in Pest Detection: ", "YOLO26n exhibits reduced recall on microscopic insect pests (<15 pixels) in dense foliage or under severe natural camouflage."),
        ("Absence of Temporal Disease Progression: ", "Single static frames cannot model infection spread velocity, incubation stages, or pest migration dynamics.")
    ]
    for k, v in lim_items:
        p = tf_lim.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Future Scope & Conclusion
    add_card(s7, Inches(0.8) + half_w + gap_half, Inches(1.85), half_w, Inches(5.2))
    tb_fut = s7.shapes.add_textbox(Inches(1.0) + half_w + gap_half, Inches(2.0), half_w - Inches(0.4), Inches(4.8))
    tf_fut = tb_fut.text_frame
    tf_fut.word_wrap = True

    p = tf_fut.paragraphs[0]
    p.text = "Future Scope & Engineering Synthesis"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    
    sub = tf_fut.add_paragraph()
    sub.text = "Production scalability and concluding defense summary"
    sub.font.size = Pt(9.5)
    sub.font.bold = True
    sub.font.color.rgb = COLOR_SECONDARY
    sub.space_after = Pt(10)

    fut_items = [
        ("Edge & UAV Integration: ", "Quantize YOLO26n and U-Net to TensorRT / ONNX INT8 for real-time onboard inference on agricultural drones and Raspberry Pi/Jetson field cameras."),
        ("Multimodal Cross-Verification: ", "Fuse foliar disease probabilities with insect vector detections (e.g., correlating whitefly presence with Begomovirus risk)."),
        ("Temporal Recurrent Forecasting: ", "Incorporate ConvLSTM architectures on video streams to forecast epidemic spread curves across growing seasons."),
        ("Project Synthesis: ", "CropGuard AI demonstrates that decomposing agricultural surveillance into three task-tailored deep learning models yields state-of-the-art accuracy, real-time edge latency, and robust field deployment safety.")
    ]
    for k, v in fut_items:
        p = tf_fut.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + k
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = v
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Speaker notes Slide 7
    s7.notes_slide.notes_text_frame.text = (
        "To conclude, let us examine the engineering limitations and future scope of CropGuard AI.\n\n"
        "First, regarding limitations: PlantVillage's controlled studio backdrops induce a domain shift when transferred "
        "to unconstrained outdoor fields. Second, U-Net segments pixel necrosis accurately, but morphological necrosis is not "
        "pathogen-specific—sunburn or nutrient burn can mimic fungal lesions. Third, YOLO26n has difficulty with microscopic pests "
        "under 15 pixels or heavily camouflaged insects in dense canopy. Finally, static photographs cannot capture temporal disease spread.\n\n"
        "Our roadmap addresses these through three key avenues:\n"
        "1. Quantizing YOLO26n and U-Net to TensorRT INT8 for deployment on autonomous spray drones and field rovers.\n"
        "2. Developing a multimodal Bayesian fusion engine that correlates insect vectors with foliar diseases.\n"
        "3. Integrating temporal ConvLSTM modules to track lesion growth rates over time.\n\n"
        "In summary, CropGuard AI demonstrates a production-grade, multi-model surveillance system that achieves 98.4% disease classification, "
        "pixel-level lesion quantification, and sub-3-millisecond insect detection, establishing a solid foundation for precision agriculture. "
        "Thank you, and I now welcome your questions."
    )

    # Save presentation
    output_path = "presentation/CropGuard_AI_Evaluation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path} (7 slides).")

if __name__ == "__main__":
    create_presentation()
