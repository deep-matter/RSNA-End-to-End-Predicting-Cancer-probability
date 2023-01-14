import matplotlib.pyplot as plt 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydicom as dc
import pandas as pf 
import numpy as np


Path_data = "../input/rsna-breast-cancer-detection/train.csv"
data_df = pd.read_csv(Path_data)
### let us get the number samples we have 
number_samples =len(data_df.patient_id)
list_col = list(data_df.columns)
print(f"number of total samples {number_samples }\n Columns dataFrame {list_col}")

"""
Notaion : we will remove the some columes thses will not use to Pipline models and Per-Processing Step 
-  ['site_id','laterality',
'view', 'age', 'biopsy',
'invasive', 'BIRADS',
'implant','machine_id',
'difficult_negative_case']
"""
df_Cols = data_df.drop(['site_id','laterality',
'view', 'age', 'biopsy',
'invasive', 'BIRADS',
'implant','machine_id',
'difficult_negative_case'],axis=1)
data_df_col = df_Cols.fillna(0) ## replace Nan with zeroes

### check labels Counts between data distrubtion 
Temp_before= data_df_col["cancer"].value_counts()
save_count_before = pd.DataFrame({ "Class":Temp_before.index,
                              "Value": Temp_before.values})
## let us figure out solution how to balance the data 
"""
solution 1 : we will try to remove some samples from data that has class 0 
and make the samples equale class 0 == class 1 
Notation : this will cause a problem is reducing the number
samples Low data repesentation to feed into the model to lean from 
"""
sub_stract_classes = save_count_before.Value.iloc[0] - save_count_before.Value.iloc[1]

px.pie(save_count_before, values="Value", names="Class", title='Cancer distribution before')

#### now we will need to sorte the data by cancer Column 
data_sorted = data_df_col.sort_values(by="cancer")
balanced_df = data_sorted[52390:]
### SAce the balnaced data set 
balanced_df.to_csv("balance.csv")

Temp_after= balanced_df["cancer"].value_counts()

save_count_after= pd.DataFrame({ "Class":Temp_after.index,
                              "Value": Temp_after.values})

fig = make_subplots(rows=1, cols=2)
Label = ["no cancer", "cancer"]
colors = ['lightslategray',] * 2
colors[1] = 'crimson'
fig.add_trace(
    go.Bar(x=Label, y=save_count_after.Value, 
           marker_color=colors,
           name="After balancing Labels"),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=Label, y=save_count_before.Value,
            marker_color=colors,
            name="Before balancing Labels"),
    row=1, col=2
)

fig.update_layout(height=500, width=700, title_text="Cancer distribution balancing ", showlegend=True)
fig.show()

Total_samples=len(balanced_df.patient_id)
print(f"the total samples we have now : {Total_samples}")


## set the seed 
np.random.seed(2000)
### here will read few samples from dataset 
Path_train = "../input/rsna-breast-cancer-detection/train_images/"
fig , axis = plt.subplots(2,2,figsize=(6,6))
rand_index = np.random.randint([900,2316],size=(1,)).item()
assing_indx = rand_index
for row in range(2):
    for col in range(2):
        ## get the full path Imae
        sub_path = str(balanced_df.patient_id.iloc[assing_indx]) + "/" + str(balanced_df.image_id.iloc[assing_indx])
        Label = balanced_df.cancer.iloc[assing_indx]
        Image_path = os.path.join(Path_train,sub_path) + ".dcm"
        Image_array = dc.read_file(Image_path).pixel_array 
        axis[row][col].imshow(Image_array,cmap="bone")
        if Label == 0:
            axis[row][col].set_title("Negative")
        else:
            axis[row][col].set_title("Positive")
    assing_indx += 1     
