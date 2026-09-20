import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from inference_model import (
    load_model,
    predict_image
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "best_multitask_resnet50_final11.pth"
)

IMAGE_FOLDER = (
    "input_images"
)

EXCEL_PATH = (
    "Flood_Vulnerability_Curves_Residential_Types (1).xlsx"
)

OUTPUT_FOLDER = (
    "results"
)


# ============================================================
# MATERIAL → BUILDING TYPE
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
# DAMAGE CLASSES
# ============================================================

DAMAGE_CLASSES = [
    "Low",
    "Moderate",
    "Severe"
]


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = load_model(
    CHECKPOINT
)

print(
    "Model loaded successfully."
)


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


image_paths = []

for filename in sorted(
    os.listdir(IMAGE_FOLDER)
):

    if filename.lower().endswith(
        image_extensions
    ):

        image_paths.append(
            os.path.join(
                IMAGE_FOLDER,
                filename
            )
        )


if len(image_paths) == 0:

    raise ValueError(
        "No images found in "
        f"'{IMAGE_FOLDER}'."
    )


print()
print("=" * 70)
print("IMAGES")
print("=" * 70)

print(
    "Number of images:",
    len(image_paths)
)


# ============================================================
# RUN MODEL ON ALL IMAGES
# ============================================================

predictions = []


print()
print("=" * 70)
print("IMAGE PREDICTIONS")
print("=" * 70)


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

    building_type = (
        MATERIAL_TO_BUILDING[
            material
        ]
    )


    prediction = {

        "Image":
            filename,

        "Material":
            material,

        "Material_Confidence":
            result[
                "material_confidence"
            ],

        "Damage":
            damage,

        "Damage_Confidence":
            result[
                "damage_confidence"
            ],

        "Building_Type":
            building_type
    }


    predictions.append(
        prediction
    )


    print()
    print(
        filename
    )

    print(
        "  Material:",
        material,
        f"({result['material_confidence']:.2f}%)"
    )

    print(
        "  Damage:",
        damage,
        f"({result['damage_confidence']:.2f}%)"
    )

    print(
        "  Building Type:",
        building_type
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    predictions
)


prediction_csv = os.path.join(
    OUTPUT_FOLDER,
    "image_predictions.csv"
)


prediction_df.to_csv(
    prediction_csv,
    index=False
)


# ============================================================
# COUNTS
# ============================================================

print()
print("=" * 70)
print("MATERIAL COUNTS")
print("=" * 70)


material_counts = Counter(
    prediction_df["Material"]
)


for material in [
    "Brick",
    "Timber",
    "Concrete"
]:

    print(
        f"{material:10s}: "
        f"{material_counts[material]}"
    )


print()
print("=" * 70)
print("DAMAGE COUNTS")
print("=" * 70)


damage_counts = Counter(
    prediction_df["Damage"]
)


for damage in DAMAGE_CLASSES:

    print(
        f"{damage:10s}: "
        f"{damage_counts[damage]}"
    )


print()
print("=" * 70)
print("BUILDING TYPE COUNTS")
print("=" * 70)


building_counts = Counter(
    prediction_df["Building_Type"]
)


for building_type in [
    "Residential Permanent",
    "Residential Semi-Permanent",
    "Residential Temporary"
]:

    print(
        f"{building_type:30s}: "
        f"{building_counts[building_type]}"
    )


# ============================================================
# LOAD EXCEL
# ============================================================

print()
print("=" * 70)
print("LOADING CURVES")
print("=" * 70)


curve_df = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Curve_Data"
)


curve_sets = sorted(
    curve_df[
        "Curve_Set"
    ].unique()
)


depths = sorted(
    curve_df[
        "Flood_Depth_m"
    ].unique()
)


# ============================================================
# FUNCTION:
# CALCULATE CURVE FOR A GROUP OF IMAGES
# ============================================================

def calculate_group_curve(
    group_df
):

    total_images = len(
        group_df
    )


    # Count predicted building types
    counts = Counter(
        group_df[
            "Building_Type"
        ]
    )


    rows = []


    for curve_set in curve_sets:

        curve_subset = curve_df[
            curve_df[
                "Curve_Set"
            ] == curve_set
        ]


        for depth in depths:

            aggregate_damage = 0.0


            # ------------------------------------------------
            # Permanent
            # ------------------------------------------------

            n_permanent = counts[
                "Residential Permanent"
            ]

            permanent_row = curve_subset[
                (
                    curve_subset[
                        "Building_Type"
                    ]
                    ==
                    "Residential Permanent"
                )
                &
                (
                    curve_subset[
                        "Flood_Depth_m"
                    ]
                    ==
                    depth
                )
            ]


            if len(
                permanent_row
            ) > 0:

                ratio = float(
                    permanent_row[
                        "Damage_Ratio"
                    ].iloc[0]
                )

                aggregate_damage += (
                    n_permanent
                    * ratio
                )


            # ------------------------------------------------
            # Semi-Permanent
            # ------------------------------------------------

            n_semi = counts[
                "Residential Semi-Permanent"
            ]

            semi_row = curve_subset[
                (
                    curve_subset[
                        "Building_Type"
                    ]
                    ==
                    "Residential Semi-Permanent"
                )
                &
                (
                    curve_subset[
                        "Flood_Depth_m"
                    ]
                    ==
                    depth
                )
            ]


            if len(
                semi_row
            ) > 0:

                ratio = float(
                    semi_row[
                        "Damage_Ratio"
                    ].iloc[0]
                )

                aggregate_damage += (
                    n_semi
                    * ratio
                )


            # ------------------------------------------------
            # Temporary
            # ------------------------------------------------

            n_temporary = counts[
                "Residential Temporary"
            ]

            temporary_row = curve_subset[
                (
                    curve_subset[
                        "Building_Type"
                    ]
                    ==
                    "Residential Temporary"
                )
                &
                (
                    curve_subset[
                        "Flood_Depth_m"
                    ]
                    ==
                    depth
                )
            ]


            if len(
                temporary_row
            ) > 0:

                ratio = float(
                    temporary_row[
                        "Damage_Ratio"
                    ].iloc[0]
                )

                aggregate_damage += (
                    n_temporary
                    * ratio
                )


            # ------------------------------------------------
            # Mean damage ratio
            # ------------------------------------------------

            if total_images > 0:

                mean_damage = (
                    aggregate_damage
                    / total_images
                )

            else:

                mean_damage = 0.0


            rows.append({

                "Curve_Set":
                    curve_set,

                "Flood_Depth_m":
                    depth,

                "Aggregate_Damage":
                    aggregate_damage,

                "Mean_Damage_Ratio":
                    mean_damage,

                "Mean_Damage_Percent":
                    mean_damage * 100,

                "Number_of_Images":
                    total_images
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# ALL IMAGES
# ============================================================

print()
print("=" * 70)
print("GENERATING ALL-IMAGE CURVES")
print("=" * 70)


all_curve_df = calculate_group_curve(
    prediction_df
)


all_curve_csv = os.path.join(
    OUTPUT_FOLDER,
    "all_classes_curves.csv"
)


all_curve_df.to_csv(
    all_curve_csv,
    index=False
)


# ============================================================
# PLOT ALL IMAGES
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for curve_set in curve_sets:

    subset = all_curve_df[
        all_curve_df[
            "Curve_Set"
        ] == curve_set
    ].sort_values(
        "Flood_Depth_m"
    )


    plt.plot(
        subset[
            "Flood_Depth_m"
        ],

        subset[
            "Mean_Damage_Percent"
        ],

        marker="o",

        label=curve_set
    )


plt.xlabel(
    "Flood Depth (m)"
)

plt.ylabel(
    "Mean Damage (%)"
)

plt.title(
    "All Classes - Aggregate Depth-Damage Curves"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    fontsize=8
)

plt.tight_layout()


all_plot = os.path.join(
    OUTPUT_FOLDER,
    "all_classes_curves.png"
)


plt.savefig(
    all_plot,
    dpi=200
)

plt.close()


print(
    "Saved:",
    all_plot
)


# ============================================================
# DAMAGE-CLASS-SPECIFIC CURVES
# ============================================================

print()
print("=" * 70)
print("GENERATING DAMAGE-CLASS CURVES")
print("=" * 70)


for damage_class in DAMAGE_CLASSES:

    class_df = prediction_df[
        prediction_df[
            "Damage"
        ]
        ==
        damage_class
    ].copy()


    if len(
        class_df
    ) == 0:

        print(
            f"No images predicted as "
            f"{damage_class}. Skipping."
        )

        continue


    print()
    print(
        f"{damage_class}: "
        f"{len(class_df)} images"
    )


    class_curve_df = calculate_group_curve(
        class_df
    )


    class_csv = os.path.join(
        OUTPUT_FOLDER,
        f"{damage_class.lower()}_curves.csv"
    )


    class_curve_df.to_csv(
        class_csv,
        index=False
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )


    for curve_set in curve_sets:

        subset = class_curve_df[
            class_curve_df[
                "Curve_Set"
            ] == curve_set
        ].sort_values(
            "Flood_Depth_m"
        )


        plt.plot(
            subset[
                "Flood_Depth_m"
            ],

            subset[
                "Mean_Damage_Percent"
            ],

            marker="o",

            label=curve_set
        )


    plt.xlabel(
        "Flood Depth (m)"
    )

    plt.ylabel(
        "Mean Damage (%)"
    )

    plt.title(
        f"{damage_class} Class - "
        "Depth-Damage Curves"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()


    plot_path = os.path.join(
        OUTPUT_FOLDER,
        f"{damage_class.lower()}_curves.png"
    )


    plt.savefig(
        plot_path,
        dpi=200
    )

    plt.close()


    print(
        "Saved:",
        plot_path
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("AGGREGATION COMPLETE")
print("=" * 70)

print()
print(
    "Prediction CSV:",
    prediction_csv
)

print(
    "All-class curve CSV:",
    all_curve_csv
)

print(
    "Output folder:",
    OUTPUT_FOLDER
)

print()
print(
    "Images processed:",
    len(prediction_df)
)

print()
print(
    "The generated curves use the Excel "
    "damage ratios."
)

print(
    "Aggregate_Damage = "
    "sum(number of images × damage ratio)"
)

print(
    "Mean_Damage_Ratio = "
    "Aggregate_Damage / number of images"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)