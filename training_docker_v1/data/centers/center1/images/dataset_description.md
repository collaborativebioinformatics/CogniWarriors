# Dataset CSVs

This folder contains MRI feature CSVs organized in three ways.

## `embeddings/`

### `train_embedding.csv`

-   59 rows
-   Columns: `subject`, `session`, `vector`
-   `vector` contains the 80-dimensional MRI feature vector
-   All samples are `ses-1`

### `test_embedding.csv`

-   164 rows
-   Columns: `subject`, `session`, `vector`
-   `vector` contains the 80-dimensional MRI feature vector
-   Contains `ses-1`, `ses-2`, and `ses-3`

## `sorted_by_session/`

### `ses1.csv`

-   130 rows
-   All `ses-1` samples
-   Contains subject/session information and the 80 MRI features

### `ses2.csv`

-   73 rows
-   All `ses-2` samples
-   Contains subject/session information and the 80 MRI features

### `ses3.csv`

-   20 rows
-   All `ses-3` samples
-   Contains subject/session information and the 80 MRI features

## `sorted_by_test_or_train/`

### `train.csv`

-   59 rows
-   Subjects with only one available session
-   All are `ses-1`
-   Contains the 80 MRI features and phenotype information

### `test.csv`

-   164 rows
-   Subjects with multiple available sessions
-   Contains `ses-1`, `ses-2`, and `ses-3`
-   Contains the 80 MRI features and phenotype information
