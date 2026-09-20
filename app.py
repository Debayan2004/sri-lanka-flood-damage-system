import os
import uuid

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from inference_model import load_model, predict_image


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "best_multitask_resnet50_final11.pth"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

PLOT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "plots"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PLOT_FOLDER,
    exist_ok=True
)


# ============================================================
# FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# MATERIAL → EXCEL BUILDING TYPE
# ============================================================

MATERIAL_TO_BUILDING = {

    "Brick":
        "Residential Permanent",

    "Timber":
        "Residential Semi-Permanent",

    "Concrete":
        "Residential Temporary"
}


# ============================================================
# EXCEL COLUMNS
# ============================================================

EXPECTED_COLUMNS = [

    "Curve_Set",
    "Building_Type",
    "Flood_Depth_m",
    "Damage_Ratio",
    "Damage_Percent"

]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = load_model(
    CHECKPOINT_PATH
)



print("Model loaded successfully.")
print()


# ============================================================
# IMAGE CHECK
# ============================================================

def is_image(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in IMAGE_EXTENSIONS


# ============================================================
# LOAD EXCEL
# ============================================================

def load_curve_excel(excel_path):

    df = pd.read_excel(
        excel_path,
        sheet_name="Curve_Data"
    )

    missing_columns = [

        column

        for column in EXPECTED_COLUMNS

        if column not in df.columns

    ]

    if missing_columns:

        raise ValueError(
            "Excel is missing required columns: "
            + str(missing_columns)
        )


    df = df.dropna(
        subset=[
            "Curve_Set",
            "Building_Type",
            "Flood_Depth_m",
            "Damage_Ratio"
        ]
    ).copy()


    df["Flood_Depth_m"] = pd.to_numeric(
        df["Flood_Depth_m"],
        errors="coerce"
    )

    df["Damage_Ratio"] = pd.to_numeric(
        df["Damage_Ratio"],
        errors="coerce"
    )

    df["Damage_Percent"] = pd.to_numeric(
        df["Damage_Percent"],
        errors="coerce"
    )


    df = df.dropna(
        subset=[
            "Flood_Depth_m",
            "Damage_Ratio"
        ]
    )


    return df


# ============================================================
# PREDICT IMAGES
# ============================================================

def predict_images(image_paths):

    predictions = []


    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )


        result = predict_image(
            model,
            image_path
        )


        material = result[
            "material"
        ]

        damage = result[
            "damage"
        ]


        if material not in MATERIAL_TO_BUILDING:

            raise ValueError(
                "Unknown material predicted: "
                + str(material)
            )


        building_type = (
            MATERIAL_TO_BUILDING[
                material
            ]
        )


        predictions.append({

            "image":
                filename,

            "material":
                material,

            "material_confidence":
                float(
                    result[
                        "material_confidence"
                    ]
                ),

            "damage":
                damage,

            "damage_confidence":
                float(
                    result[
                        "damage_confidence"
                    ]
                ),

            "building_type":
                building_type

        })


    return predictions


# ============================================================
# COUNTS
# ============================================================

def get_counts(predictions):

    material_counts = {

        "Brick": 0,
        "Timber": 0,
        "Concrete": 0

    }


    damage_counts = {

        "Low": 0,
        "Moderate": 0,
        "Severe": 0

    }


    building_counts = {

        "Residential Permanent": 0,

        "Residential Semi-Permanent": 0,

        "Residential Temporary": 0

    }


    for prediction in predictions:

        material_counts[
            prediction["material"]
        ] += 1


        damage_counts[
            prediction["damage"]
        ] += 1


        building_counts[
            prediction["building_type"]
        ] += 1


    return (
        material_counts,
        damage_counts,
        building_counts
    )


# ============================================================
# CREATE ONE CURVE-SET GRAPH
#
# IMPORTANT:
# Every graph contains the THREE original Excel curves:
#
#   Residential Permanent
#   Residential Semi-Permanent
#   Residential Temporary
#
# No Low/Moderate/Severe filtering happens here.
# ============================================================

def create_curve_set_plot(
    curve_df,
    curve_set,
    prefix,
    building_counts
):

    curve_subset = curve_df[
        curve_df["Curve_Set"] == curve_set
    ].copy()

    if curve_subset.empty:
        return None

    building_types = [
        "Residential Permanent",
        "Residential Semi-Permanent",
        "Residential Temporary"
    ]

    unique_id = uuid.uuid4().hex[:10]

    safe_prefix = (
        prefix
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    filename = f"{safe_prefix}_{unique_id}.png"

    output_path = os.path.join(
        PLOT_FOLDER,
        filename
    )

    # ========================================================
    # MAIN AXIS — ORIGINAL THREE CURVES
    # ========================================================

    fig, ax1 = plt.subplots(
        figsize=(10, 7)
    )

    for building_type in building_types:

        subset = curve_subset[
            curve_subset["Building_Type"] == building_type
        ].sort_values(
            "Flood_Depth_m"
        )

        if subset.empty:
            continue

        ax1.plot(
            subset["Flood_Depth_m"],
            subset["Damage_Ratio"],
            marker="o",
            linewidth=2.5,
            label=building_type
        )

    # ========================================================
    # TOTAL DAMAGE CALCULATION
    # ========================================================

    permanent_count = building_counts[
        "Residential Permanent"
    ]

    semi_permanent_count = building_counts[
        "Residential Semi-Permanent"
    ]

    temporary_count = building_counts[
        "Residential Temporary"
    ]

    depths = sorted(
        curve_subset["Flood_Depth_m"].unique()
    )

    total_damage = []

    for depth in depths:

        permanent_ratio = curve_subset[
            (curve_subset["Building_Type"]
             == "Residential Permanent")
            &
            (curve_subset["Flood_Depth_m"]
             == depth)
        ]["Damage_Ratio"].sum()

        semi_permanent_ratio = curve_subset[
            (curve_subset["Building_Type"]
             == "Residential Semi-Permanent")
            &
            (curve_subset["Flood_Depth_m"]
             == depth)
        ]["Damage_Ratio"].sum()

        temporary_ratio = curve_subset[
            (curve_subset["Building_Type"]
             == "Residential Temporary")
            &
            (curve_subset["Flood_Depth_m"]
             == depth)
        ]["Damage_Ratio"].sum()

        total = (
            permanent_count * permanent_ratio
            +
            semi_permanent_count * semi_permanent_ratio
            +
            temporary_count * temporary_ratio
        )

        total_damage.append(total)

    # ========================================================
    # SECONDARY AXIS — TOTAL DAMAGE
    # ========================================================

    ax2 = ax1.twinx()

    ax2.plot(
        depths,
        total_damage,
        marker="o",
        linewidth=3.0,
        linestyle="--",
        label="Total Damage (All Buildings)"
    )

    # ========================================================
    # LABELS
    # ========================================================

    ax1.set_title(
        f"{curve_set}: Residential Depth-Damage Curves",
        fontsize=16
    )

    ax1.set_xlabel(
        "Flood Depth (m)",
        fontsize=12
    )

    ax1.set_ylabel(
        "Damage Ratio",
        fontsize=12
    )

    ax2.set_ylabel(
        "Total Damage (All Buildings)",
        fontsize=12
    )

    # ========================================================
    # LEFT AXIS LIMIT
    # ========================================================

    max_original = curve_subset[
        "Damage_Ratio"
    ].max()

    ax1.set_ylim(
        0,
        max_original * 1.15
    )

    # ========================================================
    # RIGHT AXIS LIMIT
    # ========================================================

    max_total = max(
        total_damage,
        default=0
    )

    if max_total > 0:
        ax2.set_ylim(
            0,
            max_total * 1.15
        )

    # ========================================================
    # GRID
    # ========================================================

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    # ========================================================
    # COMBINED LEGEND
    # ========================================================

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right"
    )

    # ========================================================
    # SAVE
    # ========================================================

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return (
        "/static/plots/"
        + filename
    )


# ============================================================
# GENERATE ALL FIVE EXCEL GRAPHS
# ============================================================

def generate_all_curve_plots(
    curve_df,
    building_counts
):

    curve_sets = [

        "AMMA material-based adapted",

        "JRC global/Asia adapted",

        "UK/MCM-inspired provisional",

        "HAZUS-style adapted",

        "Sri Lanka/Colombo adapted"

    ]


    plots = []


    for curve_set in curve_sets:

        if curve_set not in (
            curve_df["Curve_Set"]
            .unique()
        ):

            continue


        plot_url = create_curve_set_plot(
            curve_df,
            curve_set,
            curve_set,
            building_counts
        )


        if plot_url:

            plots.append({

                "curve_set":
                    curve_set,

                "plot":
                    plot_url

            })


    return plots


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ORIGINAL /predict
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({

            "error":
                "No image uploaded."

        }), 400


    image = request.files[
        "image"
    ]


    if image.filename == "":

        return jsonify({

            "error":
                "No image selected."

        }), 400


    if not is_image(
        image.filename
    ):

        return jsonify({

            "error":
                "Unsupported image format."

        }), 400


    filename = (

        uuid.uuid4().hex
        + "_"
        + secure_filename(
            image.filename
        )

    )


    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    image.save(
        image_path
    )


    try:

        result = predict_image(
            model,
            image_path
        )


        material = result[
            "material"
        ]


        return jsonify({

            "material":
                material,

            "confidence":
                result[
                    "material_confidence"
                ],

            "damage":
                result[
                    "damage"
                ],

            "damage_confidence":
                result[
                    "damage_confidence"
                ],

            "building_type":
                MATERIAL_TO_BUILDING[
                    material
                ]

        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        images = request.files.getlist(
            "images"
        )


        images = [

            image

            for image in images

            if image
            and image.filename

        ]


        if len(images) == 0:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please upload at least one "
                    "building image."

            }), 400


        # ----------------------------------------------------
        # EXCEL
        # ----------------------------------------------------

        if "excel" not in request.files:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please upload the Excel "
                    "curve file."

            }), 400


        excel_file = request.files[
            "excel"
        ]


        if not excel_file.filename:

            return jsonify({

                "success":
                    False,

                "error":
                    "No Excel file selected."

            }), 400


        # ----------------------------------------------------
        # REQUEST FOLDER
        # ----------------------------------------------------

        request_id = uuid.uuid4().hex[:12]


        request_folder = os.path.join(

            UPLOAD_FOLDER,

            request_id

        )


        os.makedirs(
            request_folder,
            exist_ok=True
        )


        # ----------------------------------------------------
        # SAVE EXCEL
        # ----------------------------------------------------

        excel_filename = secure_filename(
            excel_file.filename
        )


        excel_path = os.path.join(

            request_folder,

            excel_filename

        )


        excel_file.save(
            excel_path
        )


        # ----------------------------------------------------
        # SAVE IMAGES
        # ----------------------------------------------------

        image_paths = []


        for index, image in enumerate(
            images
        ):

            if not is_image(
                image.filename
            ):

                continue


            safe_name = secure_filename(
                image.filename
            )


            image_filename = (

                f"{index:04d}_"
                f"{safe_name}"

            )


            image_path = os.path.join(

                request_folder,

                image_filename

            )


            image.save(
                image_path
            )


            image_paths.append(
                image_path
            )


        if len(image_paths) == 0:

            return jsonify({

                "success":
                    False,

                "error":
                    "No supported image files "
                    "were uploaded."

            }), 400


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = predict_images(
            image_paths
        )


        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        (
            material_counts,
            damage_counts,
            building_counts
        ) = get_counts(
            predictions
        )


        # ----------------------------------------------------
        # EXCEL
        # ----------------------------------------------------

        curve_df = load_curve_excel(
            excel_path
        )


        # ----------------------------------------------------
        # GENERATE THE FIVE GRAPHS
        # ----------------------------------------------------

        curve_plots = generate_all_curve_plots(
            curve_df,
            building_counts
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "predictions":
                predictions,

            "total_images":
                len(predictions),

            "material_counts":
                material_counts,

            "damage_counts":
                damage_counts,

            "building_counts":
                building_counts,

            "curve_plots":
                curve_plots

        })


    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(e))


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SRI LANKA FLOOD DAMAGE APPLICATION")
    print("=" * 70)

    print()
    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )