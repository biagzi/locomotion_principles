import numpy as np

def diversity_check(pop,new_children,diverse_children,similarity_threshold_children = 0.95,similarity_threshold_parent = 0.95):
    """ Check if the new children are diverse (comparing with themselfs and with the parents)
    Return just the children that are diverse """

    for child in new_children:
        SIMILAR = False

        child_shape = child.genotype.to_phenotype_mapping['material']['state'].astype(bool)

        for div_child in diverse_children: #check if child is diverse concerning the already diverse children
            div_child_shape = div_child.genotype.to_phenotype_mapping['material']['state'].astype(bool)
            similarity = float(np.logical_and(child_shape,div_child_shape).sum())/min(float(child_shape.sum()),float(div_child_shape.sum())) #similarity =1 if they are totally equal
            if similarity >= similarity_threshold_children: #if they are considered similar
                SIMILAR = True
                break

        while SIMILAR == False: #if the child passed through the diverse children

            for parent in pop:  #check if child is diverse concerning the old population
                parent_shape = parent.genotype.to_phenotype_mapping['material']['state'].astype(bool)
                similarity = float(np.logical_and(child_shape,parent_shape).sum())/min(float(child_shape.sum()),float(parent_shape.sum()))
                if similarity_threshold_parent <= similarity < 1.0: #to allow mutations only in phase offset
                    SIMILAR = True
                    break
            
            if SIMILAR == False:
                diverse_children.append(child)
                if len(diverse_children) == pop.pop_size:
                    return diverse_children 
                break
        
        

    return diverse_children

def diversity_check_in_children(pop,new_children,diverse_children,similarity_threshold_children = 0.95):
    """ Check if the new children are diverse (comparing with themselfs)
    Return just the children that are diverse """

    for child in new_children:
        SIMILAR = False
        child_shape = child.genotype.to_phenotype_mapping['material']['state'].astype(bool)

        for div_child in diverse_children: #check if child is diverse concerning the already diverse children
            div_child_shape = div_child.genotype.to_phenotype_mapping['material']['state'].astype(bool)
            similarity = float(np.logical_and(child_shape,div_child_shape).sum())/min(float(child_shape.sum()),float(div_child_shape.sum())) #similarity =1 if they are totally equal
            if similarity >= similarity_threshold_children:
                SIMILAR = True
                break
        
        if SIMILAR == False:
            diverse_children.append(child)
            if len(diverse_children) == pop.pop_size:
                return diverse_children 


    return diverse_children

