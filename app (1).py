import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

d = pd.read_csv("D:\\LY\\dml\\Iris.csv")

def fun(csv_file):
    print(csv_file)
    data = pd.read_csv(csv_file.name)
    return data    

def displayDetails(col):
    sum = np.sum(np.array(d.loc[:,col]))
    return sum/len(d)

def showGraph(col):
    # sum = np.sum(np.array(d.loc[:,col]))
    # avg = sum/len(d)
    # sum = 0
    # for i in range(len(d)):
    #     sum += (d.loc[i,col]-avg)*(d.loc[i,col]-avg)
    # var = sum/len(d)
    # sd = math.sqrt(var)
    # z = (np.array(d.loc[:,col])-avg)/sd
    print(col)
    fig = plt.figure()
    # stats.probplot(z, dist="norm", plot=plt)
    # sns.set_style("whitegrid")
    # ax = sns.FacetGrid(d, hue="Species", height=5).map(sns.histplot, col).add_legend()
    # sns.boxplot(x=col,y="SepalWidthCm",data=d)
    # plt.title("Boxplot")
    sns.FacetGrid(d, hue="Species", height=4).map(plt.scatter, col, "SepalWidthCm").add_legend()
    plt.title("Scatter plot")
                                                               
    # plt.title("Histogram")

    return fig

with gr.Blocks() as demo:
    with gr.Tab("Display Dataset"):
        dataset = gr.File(label="Upload Dataset")
        output = gr.Dataframe(label="Dataset")
        greet_btn = gr.Button("Submit")
        greet_btn.click(fn=fun, inputs=dataset, outputs=output)
    with gr.Tab("Question 1"):
        dropdown = gr.Dropdown(list(d.columns), label="Select Column")
        mean = gr.TextArea(label="Mean")
        display_btn = gr.Button("Submit")
        display_btn.click(fn=displayDetails, inputs=dropdown, outputs=mean)
    with gr.Tab("Graph"):
        dropdown = gr.Dropdown(list(d.columns), label="Select Column")
        output = gr.Plot()
        display_btn = gr.Button("Submit")
        display_btn.click(fn=showGraph, inputs=dropdown, outputs=output)
        
# demo = gr.Interface(fn=fun, inputs=[gr.File(label="Upload Dataset")],
#                     outputs=[gr.Dataframe(label="Dataset")],
#                     title="Sample Code")
demo.launch()