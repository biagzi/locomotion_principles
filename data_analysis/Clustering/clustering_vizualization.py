import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def print_one_morpho(po,shape,size,nrows,nlines,n,fig,id,elev=20, azim=80,matrix_label = None,nclus=None,avg_degree=None,metric=None,stiff = None, env = None,from_env = None,fit=None):
    """ Print one morpho"""

    plt.rcParams.update({'axes.titlesize': 'small'})

    ax = fig.add_subplot(nrows, nlines, n, projection='3d')

    ax.set_xlim([0, size[0]])
    ax.set_ylim([0, size[1]])
    ax.set_zlim([0, size[2]])

    po = np.matrix.round(po,3)
    norm = plt.Normalize(-1, 1)

    # shape = np.rot90(shape,k=ROTATE)
    # po = np.rot90(po,k=ROTATE)
    
    #ax.set_aspect('equal')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Set transaprency of axes
    ax.patch.set_alpha(0)

    np.random.seed(1) 
    color_pallet = sns.color_palette("Set3",nclus) + sns.color_palette("hls",nclus) + sns.color_palette("Set2",nclus) + sns.color_palette("Set1",nclus)

    if matrix_label is None: 
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):            
                    if shape[x, y, z]== 9:
                        ax.bar3d(x, y, z, 1, 1, 1, color=cm.RdBu(norm(po[x][y][z])),linewidth=0.25, edgecolor='black')
                    if shape[x,y,z]==1 :
                        ax.bar3d(x, y, z, 1, 1, 1, color='yellow',linewidth=0.25, edgecolor='black')
        plot_text = []
        text_names = ["from","id","fit",'stiff',"env","Nclus","AvgD","metric"]
        text_values = [from_env,id,fit,stiff,env,nclus,avg_degree,metric]
        for i, text_value in enumerate(text_values):
            if text_value is not None:
                if text_names[i] == "stiff":
                    plot_text.append('{0}:{1:.0e}'.format(text_names[i],text_value))
                elif text_names[i] == "fit":
                    plot_text.append('{0}:{1}'.format(text_names[i],round(text_value,1)))
                else:
                    plot_text.append('{0}:{1}'.format(text_names[i],text_value))

        ax.title.set_text("\n".join([text for text in plot_text]))
        ax.title.set_color(color = 'red')
    else:
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if shape[x, y, z]== 9 or shape[x, y, z]== 1:
                        if matrix_label[x][y][z]== -1:
                            ax.bar3d(x, y, z, 1, 1, 1, color='grey',linewidth=0.25, edgecolor='black')
                        elif matrix_label[x][y][z] >= 0:
                            ax.bar3d(x, y, z, 1, 1, 1, color=color_pallet[int(matrix_label[x][y][z])],linewidth=0.25, edgecolor='black')
        plot_text = []
        text_names = ["from","id","fit",'stiff','env',"Nclus","AvgD","metric"]
        text_values = [from_env,id,fit,stiff,env,nclus,avg_degree,metric] 
        for i, text_value in enumerate(text_values):
            if text_value is not None:
                if text_names[i] == "stiff":
                    plot_text.append('{0}:{1:.0e}'.format(text_names[i],text_value))
                elif text_names[i] == "fit":
                    try:
                        plot_text.append('{0}:{1}'.format(text_names[i],round(text_value,1)))
                    except:
                        plot_text.append('{0}:{1}'.format(text_names[i],text_value))
                else:
                    plot_text.append('{0}:{1}'.format(text_names[i],text_value))

        ax.title.set_text("\n".join([text for text in plot_text]))
        ax.title.set_color(color = 'red')

def list_to_matrix(labels,X,size):
    """ Function to transform labels list in morpho matrix """

    matrix = np.full_like(np.zeros(size),-2)
    for i in range(len(labels)):
        x,y,z,label = X[i][0], X[i][1],X[i][2],labels[i]
        matrix[x][y][z] = int(label)
        
    return matrix


def print_cluster_po_by_side(SEED,I_VALUE, NUMBER_OF_FIGS,SIZE,EXP_NAME,MAX_GEN,CLUSTERING_NAME,SAVE_FIG = False,encode='ASCII'):
    """ Function to print cluster and phase offset morphologies side by side"""

    from clustering_algorithm import processing_data_to_cluster_light_no_pt
    from clustering_utils import open_cluster_seed
    from BasicAnalysisUtils import return_shape_po_fitness
    
    #Collect data
    for seed in range(SEED,SEED+1):

        all_fits_shape_po = return_shape_po_fitness(seed,EXP_NAME,MAX_GEN,encode=encode)
        clustering_allX_results = open_cluster_seed(seed,EXP_NAME,CLUSTERING_NAME,encode = encode)
        all_X = processing_data_to_cluster_light_no_pt(seed,EXP_NAME,SIZE,MAX_GEN,encode,factor=10,return_gen = False)

    #Plot shapes
    fig = plt.figure(figsize=(15,25))
    n = 1
    for i in range(I_VALUE, I_VALUE+NUMBER_OF_FIGS):
        key = list(all_fits_shape_po)[i]
        po = all_fits_shape_po[key]['po']
        shape = all_fits_shape_po[key]['shape']
        fit = round(all_fits_shape_po[key]['fit'],1)
        print_one_morpho(po,shape,SIZE,NUMBER_OF_FIGS/2 + 1,n,fig,id =key,fit=fit) 
        
        n += 1
        n_cluster = clustering_allX_results[key]['n_clusters']
        matrix = list_to_matrix(clustering_allX_results[key]['labels'],all_X[key],SIZE)
        try: 
            metric = round(clustering_allX_results[key]['metric']['SC'],2)
        except:
            metric = 'error'
        print_one_morpho(po,shape,SIZE,NUMBER_OF_FIGS/2 + 1,n,fig,key, matrix_label = matrix, nclus = n_cluster,metric =metric)  
        n += 1
    

    fig.subplots_adjust(wspace=0, hspace=0.8)
    dpi = 600 
    if SAVE_FIG:
        plt.savefig("~/locomotion_principlesdata_analysis/exp_analysis/{0}/clustering/{0}_SnapshotCluster_gen{1}_seed{2}_{3}_{4}.png".format(EXP_NAME,MAX_GEN,seed,CLUSTERING_NAME,I_VALUE), bbox_inches='tight', dpi=int(dpi), transparent=False)
    plt.close()