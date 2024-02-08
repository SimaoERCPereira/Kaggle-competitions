{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8b71007c",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:35.619958Z",
     "iopub.status.busy": "2024-02-06T15:14:35.619130Z",
     "iopub.status.idle": "2024-02-06T15:14:36.649859Z",
     "shell.execute_reply": "2024-02-06T15:14:36.647928Z"
    },
    "papermill": {
     "duration": 1.045363,
     "end_time": "2024-02-06T15:14:36.652577",
     "exception": false,
     "start_time": "2024-02-06T15:14:35.607214",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/titanic/train.csv\n",
      "/kaggle/input/titanic/test.csv\n",
      "/kaggle/input/titanic/gender_submission.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "847e1703",
   "metadata": {
    "papermill": {
     "duration": 0.009052,
     "end_time": "2024-02-06T15:14:36.671745",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.662693",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "****Load datasets****"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5ebe6429",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.693291Z",
     "iopub.status.busy": "2024-02-06T15:14:36.692162Z",
     "iopub.status.idle": "2024-02-06T15:14:36.727368Z",
     "shell.execute_reply": "2024-02-06T15:14:36.725724Z"
    },
    "papermill": {
     "duration": 0.049709,
     "end_time": "2024-02-06T15:14:36.730644",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.680935",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\n",
    "test_df = pd.read_csv(\"/kaggle/input/titanic/test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7737cd33",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.752729Z",
     "iopub.status.busy": "2024-02-06T15:14:36.751508Z",
     "iopub.status.idle": "2024-02-06T15:14:36.781406Z",
     "shell.execute_reply": "2024-02-06T15:14:36.779948Z"
    },
    "papermill": {
     "duration": 0.044189,
     "end_time": "2024-02-06T15:14:36.784587",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.740398",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Moran, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>McCarthy, Mr. Timothy J</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17463</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Palsson, Master. Gosta Leonard</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>349909</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>347742</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>237736</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "5            6         0       3   \n",
       "6            7         0       1   \n",
       "7            8         0       3   \n",
       "8            9         1       3   \n",
       "9           10         1       2   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "5                                   Moran, Mr. James    male   NaN      0   \n",
       "6                            McCarthy, Mr. Timothy J    male  54.0      0   \n",
       "7                     Palsson, Master. Gosta Leonard    male   2.0      3   \n",
       "8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   \n",
       "9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  \n",
       "5      0            330877   8.4583   NaN        Q  \n",
       "6      0             17463  51.8625   E46        S  \n",
       "7      1            349909  21.0750   NaN        S  \n",
       "8      2            347742  11.1333   NaN        S  \n",
       "9      0            237736  30.0708   NaN        C  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3f2540d8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.810553Z",
     "iopub.status.busy": "2024-02-06T15:14:36.810109Z",
     "iopub.status.idle": "2024-02-06T15:14:36.832314Z",
     "shell.execute_reply": "2024-02-06T15:14:36.830584Z"
    },
    "papermill": {
     "duration": 0.038411,
     "end_time": "2024-02-06T15:14:36.834891",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.796480",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>897</td>\n",
       "      <td>3</td>\n",
       "      <td>Svensson, Mr. Johan Cervin</td>\n",
       "      <td>male</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7538</td>\n",
       "      <td>9.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>898</td>\n",
       "      <td>3</td>\n",
       "      <td>Connolly, Miss. Kate</td>\n",
       "      <td>female</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330972</td>\n",
       "      <td>7.6292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>899</td>\n",
       "      <td>2</td>\n",
       "      <td>Caldwell, Mr. Albert Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>248738</td>\n",
       "      <td>29.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>900</td>\n",
       "      <td>3</td>\n",
       "      <td>Abrahim, Mrs. Joseph (Sophie Halaut Easu)</td>\n",
       "      <td>female</td>\n",
       "      <td>18.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2657</td>\n",
       "      <td>7.2292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>901</td>\n",
       "      <td>3</td>\n",
       "      <td>Davies, Mr. John Samuel</td>\n",
       "      <td>male</td>\n",
       "      <td>21.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>A/4 48871</td>\n",
       "      <td>24.1500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "5          897       3                    Svensson, Mr. Johan Cervin    male   \n",
       "6          898       3                          Connolly, Miss. Kate  female   \n",
       "7          899       2                  Caldwell, Mr. Albert Francis    male   \n",
       "8          900       3     Abrahim, Mrs. Joseph (Sophie Halaut Easu)  female   \n",
       "9          901       3                       Davies, Mr. John Samuel    male   \n",
       "\n",
       "    Age  SibSp  Parch     Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0     330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0     363272   7.0000   NaN        S  \n",
       "2  62.0      0      0     240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0     315154   8.6625   NaN        S  \n",
       "4  22.0      1      1    3101298  12.2875   NaN        S  \n",
       "5  14.0      0      0       7538   9.2250   NaN        S  \n",
       "6  30.0      0      0     330972   7.6292   NaN        Q  \n",
       "7  26.0      1      1     248738  29.0000   NaN        S  \n",
       "8  18.0      0      0       2657   7.2292   NaN        C  \n",
       "9  21.0      2      0  A/4 48871  24.1500   NaN        S  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd5eba86",
   "metadata": {
    "papermill": {
     "duration": 0.009905,
     "end_time": "2024-02-06T15:14:36.855443",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.845538",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Data preparation/cleasing**\n",
    "\n",
    "Check null values, duplicates, feature selection, outliers, feature engeneering, feature scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "be84791b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.878541Z",
     "iopub.status.busy": "2024-02-06T15:14:36.878042Z",
     "iopub.status.idle": "2024-02-06T15:14:36.892400Z",
     "shell.execute_reply": "2024-02-06T15:14:36.890868Z"
    },
    "papermill": {
     "duration": 0.029861,
     "end_time": "2024-02-06T15:14:36.895853",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.865992",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    891\n",
       "Survived       891\n",
       "Pclass         891\n",
       "Name           891\n",
       "Sex            891\n",
       "Age            714\n",
       "SibSp          891\n",
       "Parch          891\n",
       "Ticket         891\n",
       "Fare           891\n",
       "Cabin          204\n",
       "Embarked       889\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "208fbd4d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.919162Z",
     "iopub.status.busy": "2024-02-06T15:14:36.918582Z",
     "iopub.status.idle": "2024-02-06T15:14:36.930420Z",
     "shell.execute_reply": "2024-02-06T15:14:36.929054Z"
    },
    "papermill": {
     "duration": 0.026414,
     "end_time": "2024-02-06T15:14:36.932885",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.906471",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3c879e6c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.957130Z",
     "iopub.status.busy": "2024-02-06T15:14:36.956693Z",
     "iopub.status.idle": "2024-02-06T15:14:36.967735Z",
     "shell.execute_reply": "2024-02-06T15:14:36.966349Z"
    },
    "papermill": {
     "duration": 0.02615,
     "end_time": "2024-02-06T15:14:36.970410",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.944260",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7ed94fd5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:36.994224Z",
     "iopub.status.busy": "2024-02-06T15:14:36.993722Z",
     "iopub.status.idle": "2024-02-06T15:14:38.914217Z",
     "shell.execute_reply": "2024-02-06T15:14:38.913227Z"
    },
    "papermill": {
     "duration": 1.935836,
     "end_time": "2024-02-06T15:14:38.916957",
     "exception": false,
     "start_time": "2024-02-06T15:14:36.981121",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "# For numerical columns\n",
    "\n",
    "# Initialize the imputer with your chosen strategy (mean, median, etc.)\n",
    "imputer = SimpleImputer(strategy='mean')  # Using median as an example\n",
    "\n",
    "# Fit the imputer on the training data\n",
    "# Ensure you're selecting the column as a DataFrame to maintain the 2D shape\n",
    "imputer.fit(train_df[['Fare']])\n",
    "\n",
    "# Transform both training and test datasets using the fitted imputer\n",
    "# This replaces missing values in 'Fare' based on the median of 'Fare' in the training set\n",
    "train_df['Fare'] = imputer.transform(train_df[['Fare']])\n",
    "test_df['Fare'] = imputer.transform(test_df[['Fare']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cc75a26e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:38.943966Z",
     "iopub.status.busy": "2024-02-06T15:14:38.942218Z",
     "iopub.status.idle": "2024-02-06T15:14:38.955916Z",
     "shell.execute_reply": "2024-02-06T15:14:38.954681Z"
    },
    "papermill": {
     "duration": 0.029822,
     "end_time": "2024-02-06T15:14:38.959390",
     "exception": false,
     "start_time": "2024-02-06T15:14:38.929568",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f06b5ef0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:38.985334Z",
     "iopub.status.busy": "2024-02-06T15:14:38.984461Z",
     "iopub.status.idle": "2024-02-06T15:14:38.992348Z",
     "shell.execute_reply": "2024-02-06T15:14:38.991491Z"
    },
    "papermill": {
     "duration": 0.022849,
     "end_time": "2024-02-06T15:14:38.994933",
     "exception": false,
     "start_time": "2024-02-06T15:14:38.972084",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Remove non numerical features \n",
    "\n",
    "train_df = train_df.drop (columns = ['Name','Ticket','Cabin','Embarked'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "72c79a7c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:39.022379Z",
     "iopub.status.busy": "2024-02-06T15:14:39.021751Z",
     "iopub.status.idle": "2024-02-06T15:14:39.036345Z",
     "shell.execute_reply": "2024-02-06T15:14:39.035299Z"
    },
    "papermill": {
     "duration": 0.031008,
     "end_time": "2024-02-06T15:14:39.039024",
     "exception": false,
     "start_time": "2024-02-06T15:14:39.008016",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df = pd.get_dummies(train_df, columns=['Sex'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fc8c94c5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:39.063901Z",
     "iopub.status.busy": "2024-02-06T15:14:39.063397Z",
     "iopub.status.idle": "2024-02-06T15:14:39.089201Z",
     "shell.execute_reply": "2024-02-06T15:14:39.087628Z"
    },
    "papermill": {
     "duration": 0.042222,
     "end_time": "2024-02-06T15:14:39.092189",
     "exception": false,
     "start_time": "2024-02-06T15:14:39.049967",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare  Sex_female  \\\n",
       "0              1         0       3  22.0      1      0   7.2500       False   \n",
       "1              2         1       1  38.0      1      0  71.2833        True   \n",
       "2              3         1       3  26.0      0      0   7.9250        True   \n",
       "3              4         1       1  35.0      1      0  53.1000        True   \n",
       "4              5         0       3  35.0      0      0   8.0500       False   \n",
       "..           ...       ...     ...   ...    ...    ...      ...         ...   \n",
       "886          887         0       2  27.0      0      0  13.0000       False   \n",
       "887          888         1       1  19.0      0      0  30.0000        True   \n",
       "888          889         0       3   NaN      1      2  23.4500        True   \n",
       "889          890         1       1  26.0      0      0  30.0000       False   \n",
       "890          891         0       3  32.0      0      0   7.7500       False   \n",
       "\n",
       "     Sex_male  \n",
       "0        True  \n",
       "1       False  \n",
       "2       False  \n",
       "3       False  \n",
       "4        True  \n",
       "..        ...  \n",
       "886      True  \n",
       "887     False  \n",
       "888     False  \n",
       "889      True  \n",
       "890      True  \n",
       "\n",
       "[891 rows x 9 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4d618f26",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:39.117271Z",
     "iopub.status.busy": "2024-02-06T15:14:39.116849Z",
     "iopub.status.idle": "2024-02-06T15:14:40.325305Z",
     "shell.execute_reply": "2024-02-06T15:14:40.323989Z"
    },
    "papermill": {
     "duration": 1.225039,
     "end_time": "2024-02-06T15:14:40.328584",
     "exception": false,
     "start_time": "2024-02-06T15:14:39.103545",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1000x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Assuming df is your DataFrame and 'target' is the name of your target variable\n",
    "correlation_matrix = train_df.corr()\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns  # Seaborn is used for the heatmap\n",
    "\n",
    "# Optional: Use seaborn style for the plot\n",
    "sns.set(style=\"white\")\n",
    "\n",
    "# Create a figure and a set of subplots\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "# Plotting the heatmap\n",
    "# annot=True to print the values inside the square\n",
    "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=.5)\n",
    "\n",
    "plt.title('Correlation Matrix of Variables')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35c948e3",
   "metadata": {
    "papermill": {
     "duration": 0.014345,
     "end_time": "2024-02-06T15:14:40.357163",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.342818",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Prepare datasets**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "088488cf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.389350Z",
     "iopub.status.busy": "2024-02-06T15:14:40.388885Z",
     "iopub.status.idle": "2024-02-06T15:14:40.396208Z",
     "shell.execute_reply": "2024-02-06T15:14:40.395043Z"
    },
    "papermill": {
     "duration": 0.026302,
     "end_time": "2024-02-06T15:14:40.398686",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.372384",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Feature selection\n",
    "y = train_df[\"Survived\"]\n",
    "\n",
    "features = [\"Pclass\", \"Sex_female\",'Sex_male', \"SibSp\", \"Parch\", 'Fare']\n",
    "\n",
    "X = train_df[features]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5e83359c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.429853Z",
     "iopub.status.busy": "2024-02-06T15:14:40.429328Z",
     "iopub.status.idle": "2024-02-06T15:14:40.442967Z",
     "shell.execute_reply": "2024-02-06T15:14:40.441670Z"
    },
    "papermill": {
     "duration": 0.033101,
     "end_time": "2024-02-06T15:14:40.445681",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.412580",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Encoding variable sex for the model\n",
    "test_df = pd.get_dummies(test_df, columns=['Sex'])\n",
    "\n",
    "#Feature selection\n",
    "features = [\"Pclass\", \"Sex_female\",\"Sex_male\", \"SibSp\", \"Parch\",'Fare']\n",
    "\n",
    "X_test = test_df[features]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82ab43b7",
   "metadata": {
    "papermill": {
     "duration": 0.012844,
     "end_time": "2024-02-06T15:14:40.471781",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.458937",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "****Training and testing the model in a training and validation dataset****"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fa659ae5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.501675Z",
     "iopub.status.busy": "2024-02-06T15:14:40.501126Z",
     "iopub.status.idle": "2024-02-06T15:14:40.632053Z",
     "shell.execute_reply": "2024-02-06T15:14:40.630529Z"
    },
    "papermill": {
     "duration": 0.150138,
     "end_time": "2024-02-06T15:14:40.635421",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.485283",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0628a9f4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.665819Z",
     "iopub.status.busy": "2024-02-06T15:14:40.665247Z",
     "iopub.status.idle": "2024-02-06T15:14:40.675872Z",
     "shell.execute_reply": "2024-02-06T15:14:40.674110Z"
    },
    "papermill": {
     "duration": 0.029008,
     "end_time": "2024-02-06T15:14:40.678785",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.649777",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b0b43abe",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.706826Z",
     "iopub.status.busy": "2024-02-06T15:14:40.706336Z",
     "iopub.status.idle": "2024-02-06T15:14:40.725226Z",
     "shell.execute_reply": "2024-02-06T15:14:40.724034Z"
    },
    "papermill": {
     "duration": 0.035978,
     "end_time": "2024-02-06T15:14:40.727764",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.691786",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>331</th>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>733</th>\n",
       "      <td>2</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>382</th>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>704</th>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>813</th>\n",
       "      <td>3</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>31.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>106</th>\n",
       "      <td>3</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.6500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>270</th>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>31.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>860</th>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>14.1083</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435</th>\n",
       "      <td>1</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>120.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>77.2875</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>712 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex_female  Sex_male  SibSp  Parch      Fare\n",
       "331       1       False      True      0      0   28.5000\n",
       "733       2       False      True      0      0   13.0000\n",
       "382       3       False      True      0      0    7.9250\n",
       "704       3       False      True      1      0    7.8542\n",
       "813       3        True     False      4      2   31.2750\n",
       "..      ...         ...       ...    ...    ...       ...\n",
       "106       3        True     False      0      0    7.6500\n",
       "270       1       False      True      0      0   31.0000\n",
       "860       3       False      True      2      0   14.1083\n",
       "435       1        True     False      1      2  120.0000\n",
       "102       1       False      True      0      1   77.2875\n",
       "\n",
       "[712 rows x 6 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9822e3dc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.757208Z",
     "iopub.status.busy": "2024-02-06T15:14:40.756085Z",
     "iopub.status.idle": "2024-02-06T15:14:40.762362Z",
     "shell.execute_reply": "2024-02-06T15:14:40.761535Z"
    },
    "papermill": {
     "duration": 0.023757,
     "end_time": "2024-02-06T15:14:40.764906",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.741149",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# from sklearn.model_selection import GridSearchCV\n",
    "# from sklearn.ensemble import RandomForestClassifier\n",
    "# import numpy as np\n",
    "\n",
    "# # Define the parameter grid\n",
    "# param_grid = {\n",
    "#     'n_estimators': [100, 200, 300],  # Number of trees in the forest\n",
    "#     'max_depth': [None,5, 10, 20, 30],  # Maximum number of levels in tree\n",
    "#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node\n",
    "#     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node\n",
    "#     'bootstrap': [True, False]  # Method of selecting samples for training each tree\n",
    "# }\n",
    "\n",
    "# # Initialize the classifier\n",
    "# rf = RandomForestClassifier()\n",
    "\n",
    "# # Initialize the GridSearchCV object\n",
    "# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)\n",
    "\n",
    "# # Fit the grid search to the data\n",
    "# grid_search.fit(X_train, y_train)\n",
    "\n",
    "# # Best parameters found\n",
    "# print(\"Best parameters:\", grid_search.best_params_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "fb363e43",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:40.794087Z",
     "iopub.status.busy": "2024-02-06T15:14:40.793273Z",
     "iopub.status.idle": "2024-02-06T15:14:41.123371Z",
     "shell.execute_reply": "2024-02-06T15:14:41.121526Z"
    },
    "papermill": {
     "duration": 0.348585,
     "end_time": "2024-02-06T15:14:41.126896",
     "exception": false,
     "start_time": "2024-02-06T15:14:40.778311",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(max_depth=10, min_samples_split=10, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(max_depth=10, min_samples_split=10, random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(max_depth=10, min_samples_split=10, random_state=42)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initialize the model\n",
    "rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42, min_samples_leaf= 1, min_samples_split = 10)\n",
    "\n",
    "# Train the model on the training data\n",
    "rf_clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "84853b53",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:41.161718Z",
     "iopub.status.busy": "2024-02-06T15:14:41.161192Z",
     "iopub.status.idle": "2024-02-06T15:14:41.184436Z",
     "shell.execute_reply": "2024-02-06T15:14:41.183006Z"
    },
    "papermill": {
     "duration": 0.045283,
     "end_time": "2024-02-06T15:14:41.188276",
     "exception": false,
     "start_time": "2024-02-06T15:14:41.142993",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of the Random Forest model: 0.79\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set\n",
    "y_pred = rf_clf.predict(X_val)\n",
    "\n",
    "# Calculate the accuracy of the model\n",
    "accuracy = accuracy_score(y_val, y_pred)\n",
    "print(f\"Accuracy of the Random Forest model: {accuracy:.2f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8f99bd6",
   "metadata": {
    "papermill": {
     "duration": 0.015178,
     "end_time": "2024-02-06T15:14:41.219794",
     "exception": false,
     "start_time": "2024-02-06T15:14:41.204616",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "****Final Predictions****"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "edb5db8e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:41.252150Z",
     "iopub.status.busy": "2024-02-06T15:14:41.251737Z",
     "iopub.status.idle": "2024-02-06T15:14:41.273855Z",
     "shell.execute_reply": "2024-02-06T15:14:41.272717Z"
    },
    "papermill": {
     "duration": 0.042004,
     "end_time": "2024-02-06T15:14:41.277033",
     "exception": false,
     "start_time": "2024-02-06T15:14:41.235029",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "predictions = rf_clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b5e19329",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-06T15:14:41.309899Z",
     "iopub.status.busy": "2024-02-06T15:14:41.309438Z",
     "iopub.status.idle": "2024-02-06T15:14:41.320309Z",
     "shell.execute_reply": "2024-02-06T15:14:41.318975Z"
    },
    "papermill": {
     "duration": 0.030259,
     "end_time": "2024-02-06T15:14:41.323098",
     "exception": false,
     "start_time": "2024-02-06T15:14:41.292839",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your submission was successfully saved!\n"
     ]
    }
   ],
   "source": [
    "output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})\n",
    "output.to_csv('submission.csv', index=False)\n",
    "print(\"Your submission was successfully saved!\")"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 26502,
     "sourceId": 3136,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 30646,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 10.243734,
   "end_time": "2024-02-06T15:14:42.164975",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-02-06T15:14:31.921241",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
