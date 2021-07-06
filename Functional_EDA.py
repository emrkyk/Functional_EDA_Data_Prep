import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============
# GENERAL
# =============

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}  -->', cat_cols)
    print(f'num_cols: {len(num_cols)}  -->', num_cols)
    print(f'cat_but_car: {len(cat_but_car)}  -->', cat_but_car)
    print(
        f'num_but_cat: {len(num_but_cat)}    <---   (already included in "cat_cols". Just given for reporting purposes)')

    print("{cat_cols + num_cols + cat_but_car = all variables}")

    # cat_cols + num_cols + cat_but_car = number of variables.
    # all variables: cat_cols + num_cols + cat_but_car
    # num_but_cat is included "cat_cols".
    # num_but_cat is just given for reporting purposes.

    return cat_cols, cat_but_car, num_cols, num_but_cat


# ===============================
# CATEGORICAL VARIABLES ANALYSIS
# ===============================

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("===================")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def cat_summary_adv(dataframe, categorical_cols, number_of_classes=10):
    col_count = 0
    cols_more_classes = []
    for col in categorical_cols:
        if dataframe[col].nunique() <= number_of_classes:
            print(pd.DataFrame({col: dataframe[col].value_counts(),
                                "Ratio (%)": round(100 * dataframe[col].value_counts() / len(dataframe), 2)}),
                  end="\n\n\n")
            col_count += 1
        else:
            cols_more_classes.append(dataframe[col].name)

    print(f"{col_count} categorical variables have been described.\n")
    if len(cols_more_classes) > 0:
        print(f"There are {len(cols_more_classes)} variables which have more than {number_of_classes} classes:")
        print(cols_more_classes)


# ===============================
# NUMERICAL VARIABLES ANALYSIS
# ===============================

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=30, figsize=(12, 12), density=False)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def num_hist_boxplot(dataframe, numeric_col):
    col_counter = 0
    for col in numeric_col:
        dataframe[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        sns.boxplot(x=dataframe[col], data=dataframe)
        plt.show()
        col_counter += 1
    print(f"{col_counter} variables have been plotted")


# ===============================
# TARGET VARIABLE ANALYSIS
# ===============================

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# ====================================================
# Correlations between Target and Independent Variables
# ====================================================

def find_correlation(dataframe, numeric_cols, target, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == target:
            pass
        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

# low_corrs, high_corrs = find_correlation(df, num_cols, "TARGET")


def correlation_heatmap(dataframe):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(dataframe.corr(), annot=True, cmap=colormap)
    plt.show()


