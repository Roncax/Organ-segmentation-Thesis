import numpy as np

def combine_predictions(output_masks, threshold = None):
    """Combine the output masks in one single dimension and threshold it. 
    The returned matrix has value in range (1, shape(output_mask)[0])

    Args:
        output_masks (Cxnxn numpy matrix): Output nets matrix

    Returns:
        (nxn) numpy matrix: A combination of all output mask in the first dimension of the matrix
    """

        
    matrix_shape = np.shape(output_masks[0])
    combination_matrix = np.zeros(shape=matrix_shape)
    
    output_masks[not np.argmax(output_masks)] = 0
    output_masks[output_masks >= threshold] = 1
    output_masks[output_masks < threshold] = 0

    for i in range(np.shape(output_masks)[0]):
        combination_matrix[output_masks[i,:,:] == 1] = i+1 #on single dimension - single image
        #full_output_mask[i, full_output_mask[i, :, :] == 1] = i+1 # on multiple dimension - multiple images 

    return combination_matrix
