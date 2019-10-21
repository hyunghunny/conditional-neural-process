import matplotlib.pyplot as plt

def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    """Plots the predicted mean and variance and the context points.
    
    Args: 
        target_x: An array of shape batchsize x number_targets x 1 that contains the
                x values of the target points.
        target_y: An array of shape batchsize x number_targets x 1 that contains the
                y values of the target points.
        context_x: An array of shape batchsize x number_context x 1 that contains 
                the x values of the context points.
        context_y: An array of shape batchsize x number_context x 1 that contains 
                the y values of the context points.
        pred_y: An array of shape batchsize x number_targets x 1    that contains the
                predicted means of the y values at the target points in target_x.
        pred_y: An array of shape batchsize x number_targets x 1    that contains the
                predicted variance of the y values at the target points in target_x.
    """
    # Plot everything
    plt.grid()
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k--', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ro', markersize=10)
    plt.fill_between(
            target_x[0, :, 0],
            pred_y[0, :, 0] - var[0, :, 0],
            pred_y[0, :, 0] + var[0, :, 0],
            alpha=0.2,
            facecolor='orange',
            interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.show()
