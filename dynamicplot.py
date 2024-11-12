
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle
import pandas as pd
import os


def get_zoom_boundaries():
    while True:
        try:
            # Ask for the four floating point numbers
            x_min = float(input("Enter x_min: "))
            x_max = float(input("Enter x_max: "))
            y_min = float(input("Enter y_min: "))
            y_max = float(input("Enter y_max: "))
            
            # Display the entered values for confirmation
            print(f"\nYou entered the following boundaries:")
            print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
            
            # Ask for confirmation
            confirm = input("Are these values correct? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\nBoundaries confirmed!")
                return x_min, x_max, y_min, y_max
            elif confirm == 'n':
                print("\nLet's try again...\n")
            else:
                print("\nInvalid input. Please enter 'y' for yes or 'n' for no.\n")
        
        except ValueError:
            print("\nInvalid input. Please enter valid floating point numbers.\n")


if __name__ == "__main__":


    # Initialize figure and subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    fig2, axes2 = plt.subplots(4, 2, figsize=(10, 10))
    fig2.subplots_adjust(hspace=0.4, wspace=0.4)

    datasets = {
        1: "Abalone",
        2: "Auto MPG",
        3: "California Housing",
        4: "Delta Ailerons",
        5: "LA Ozone",
        6: "Machine CPU",
        7: "Prostate Cancer",
        8: "Servo"
    }
    folder_path = os.path.join('results', 'datasets')
    df_times = pd.DataFrame(columns=["dataset", "total_timing_ELM", "mean_loop_timing_ELM", "timing_approximated_ENRELM", "timing_incremental_ENRELM"])

    for key, value in datasets.items():

        name = value

        results = np.load(os.path.join(folder_path, name + "_results.npz"))



        # Plot figures
        ax = axes[(key-1)//2, (key-1)%2]
        ax2 = axes2[(key-1)//2, (key-1)%2]
        
        X_axis = np.arange(0, results['training_error_ELM'].shape[0]+1, 1)
        # Training error plots (existing code)
        ax.fill_between(X_axis, np.concatenate([np.array([1]), results["min_training_error_ELM"]]), np.concatenate([np.array([1]), results["max_training_error_ELM"]]), color='blue', alpha=0.2)
        ax.plot(X_axis, np.concatenate([np.array([1]), results["training_error_ELM"]]), 'b-', label='ELM')
        ax.plot(X_axis, np.concatenate([np.array([1]), results['training_error_approximated_ENRELM'][0:results['training_error_ELM'].shape[0]]]), 'r-', label='A-ENR-ELM')
        non_zeros_incremental_ENRELM_training = results['training_error_incremental_ENRELM'][results['training_error_incremental_ENRELM'] != 0]
        filled_incremental_ENRELM_training = np.ones(results['training_error_ELM'].shape[0] +1 - non_zeros_incremental_ENRELM_training.shape[0]) * non_zeros_incremental_ENRELM_training[-1]
        ax.plot(X_axis[0:non_zeros_incremental_ENRELM_training.shape[0]], non_zeros_incremental_ENRELM_training, 'g-', label='I-ENR-ELM')
        ax.plot(X_axis[non_zeros_incremental_ENRELM_training.shape[0]:], filled_incremental_ENRELM_training, 'g--')
        ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1])

        fig_temp, ax_temp = plt.subplots()
        ax_temp.fill_between(X_axis, np.concatenate([np.array([1]), results["min_training_error_ELM"]]), np.concatenate([np.array([1]), results["max_training_error_ELM"]]), color='blue', alpha=0.2)
        ax_temp.plot(X_axis, np.concatenate([np.array([1]), results["training_error_ELM"]]), 'b-', label='ELM')
        ax_temp.plot(X_axis, np.concatenate([np.array([1]), results['training_error_approximated_ENRELM'][0:results['training_error_ELM'].shape[0]]]), 'r-', label='A-ENR-ELM')
        ax_temp.plot(X_axis[0:non_zeros_incremental_ENRELM_training.shape[0]], non_zeros_incremental_ENRELM_training, 'g-', label='I-ENR-ELM')
        ax_temp.plot(X_axis[non_zeros_incremental_ENRELM_training.shape[0]:], filled_incremental_ENRELM_training, 'g--')
        ax_temp.set_ylim(ax_temp.get_ylim()[0],ax_temp.get_ylim()[1])
        fig_temp.show()
        # Is zoom needed?
        user_input = input("Do you need a zoom for this training error plot?\nPlease enter 'y'/'Y' for Yes or 'n'/'N' for No: ").strip().lower()
        zoom_needed = None  # Variable to hold the True or False value
        while user_input not in ["y", 'n']:
            print("Invalid input. Please enter 'y', 'Y', 'n', or 'N'.")
            user_input = input("Please enter 'y'/'Y' for Yes or 'n'/'N': ").strip().lower()
        zoom_needed = (user_input == 'y')

        if zoom_needed:

            x_min, x_max, y_min, y_max = get_zoom_boundaries()
            plt.close(fig_temp)
        

            # Define the zoomed region
            zoom_range = (x_min, x_max)  # Adjust the range as needed
            ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')  # Inset plot
            ax_inset.fill_between(X_axis, np.concatenate([np.array([1]), results["min_training_error_ELM"]]), np.concatenate([np.array([1]), results["max_training_error_ELM"]]), color='blue', alpha=0.2)
            ax_inset.plot(X_axis, np.concatenate([np.array([1]), results["training_error_ELM"]]), 'b-')
            ax_inset.plot(X_axis, np.concatenate([np.array([1]), results['training_error_approximated_ENRELM'][0:results['training_error_ELM'].shape[0]]]), 'r-')
            ax_inset.plot(X_axis[0:non_zeros_incremental_ENRELM_training.shape[0]], non_zeros_incremental_ENRELM_training, 'g-')
            ax_inset.plot(X_axis[non_zeros_incremental_ENRELM_training.shape[0]:], filled_incremental_ENRELM_training, 'g--')
            ax_inset.set_xlim(zoom_range)  # Set x-axis limits for the inset plot

            inner_plot_ylim_max = np.maximum( np.maximum(np.max(np.concatenate([np.array([1]), results["training_error_ELM"]])[int(np.floor(x_min)): int(np.ceil(x_max))]), np.max(results['training_error_approximated_ENRELM'][0:results['training_error_ELM'].shape[0]+1][int(np.floor(x_min)): int(np.ceil(x_max))])), np.max(np.concatenate((non_zeros_incremental_ENRELM_training, filled_incremental_ENRELM_training))[int(np.floor(x_min)): int(np.ceil(x_max))]) )
            inner_plot_ylim_min = np.minimum( np.minimum(np.min(np.concatenate([np.array([1]), results["training_error_ELM"]])[int(np.floor(x_min)): int(np.ceil(x_max))]), np.min(results['training_error_approximated_ENRELM'][0:results['training_error_ELM'].shape[0]+1][int(np.floor(x_min)): int(np.ceil(x_max))])), np.min(np.concatenate((non_zeros_incremental_ENRELM_training, filled_incremental_ENRELM_training))[int(np.floor(x_min)): int(np.ceil(x_max))]) ) - 0.005

            ax_inset.set_ylim((inner_plot_ylim_min,inner_plot_ylim_max))  # Optional: match y-axis limits
            ax_inset.grid(True)

            # Draw a rectangle on the main plot to highlight the zoomed region
            rect = Rectangle((zoom_range[0], ax_inset.get_ylim()[0]), zoom_range[1] - zoom_range[0], ax_inset.get_ylim()[1] - ax_inset.get_ylim()[0],
                            edgecolor='grey', linestyle='-', linewidth=1, facecolor='none',zorder=10)
            ax.add_patch(rect)

            # Coordinates of rectangle corners on the main plot
            rect_coords_main = [
                (zoom_range[0], ax_inset.get_ylim()[0]),  # Bottom-left
                (zoom_range[1], ax_inset.get_ylim()[0]),  # Bottom-right
                (zoom_range[0], ax_inset.get_ylim()[1]),  # Top-left
                (zoom_range[1], ax_inset.get_ylim()[1])   # Top-right
            ]

            # Coordinates of the inset corners (bottom-left and top-right corners)
            inset_coords = [
                (0, 0),                             # Bottom-left
                (1, 0),                             # Bottom-right
                (0, 1),                             # Top-left
                (1, 1)                              # Top-right
            ]

            # Add connection lines for each corner
            for main_coord, inset_coord in zip(rect_coords_main, inset_coords):
                con = ConnectionPatch(xyA=inset_coord, xyB=main_coord, coordsA="axes fraction", coordsB="data", 
                                    axesA=ax_inset, axesB=ax, color="grey", linestyle="--", lw=0.5)
                ax.add_patch(con)

        else: 
            plt.close(fig_temp)

        # Test error plots with similar insets (new code for ax2)
        ax2.fill_between(X_axis, np.concatenate([np.array([1]), results["min_test_error_ELM"]]), np.concatenate([np.array([1]), results["max_test_error_ELM"]]), color='blue', alpha=0.2)
        ax2.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_ELM']]), 'b-', label='ELM')
        ax2.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_approximated_ENRELM'][0:results['test_error_ELM'].shape[0]]]), 'r-', label='A-ENR-ELM')
        non_zeros_incremental_ENRELM_test = results['test_error_incremental_ENRELM'][results['test_error_incremental_ENRELM'] != 0]
        filled_incremental_ENRELM_test = np.ones(results['test_error_ELM'].shape[0] +1 - non_zeros_incremental_ENRELM_test.shape[0]) * non_zeros_incremental_ENRELM_test[-1]
        ax2.plot(X_axis[0:non_zeros_incremental_ENRELM_test.shape[0]], non_zeros_incremental_ENRELM_test, 'g-', label = "I-ENR-ELM")
        ax2.plot(X_axis[non_zeros_incremental_ENRELM_test.shape[0]:], filled_incremental_ENRELM_test, 'g--')
        ax2.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1])

        fig_temp, ax_temp = plt.subplots()
        ax_temp.fill_between(X_axis, np.concatenate([np.array([1]), results["min_test_error_ELM"]]), np.concatenate([np.array([1]), results["max_test_error_ELM"]]), color='blue', alpha=0.2)
        ax_temp.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_ELM']]), 'b-', label='ELM')
        ax_temp.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_approximated_ENRELM'][0:results['test_error_ELM'].shape[0]]]), 'r-', label='A-ENR-ELM')
        ax_temp.plot(X_axis[0:non_zeros_incremental_ENRELM_test.shape[0]], non_zeros_incremental_ENRELM_test, 'g-', label='I-ENR-ELM')
        ax_temp.plot(X_axis[non_zeros_incremental_ENRELM_test.shape[0]:], filled_incremental_ENRELM_test, 'g--')
        ax_temp.set_ylim(ax_temp.get_ylim()[0],ax_temp.get_ylim()[1])
        fig_temp.show()
        # Is zoom needed?
        user_input = input("Do you need a zoom for this test error plot?\nPlease enter 'y'/'Y' for Yes or 'n'/'N' for No: ").strip().lower()
        zoom_needed = None  # Variable to hold the True or False value
        while user_input not in ["y", 'n']:
            print("Invalid input. Please enter 'y', 'Y', 'n', or 'N'.")
            user_input = input("Please enter 'y'/'Y' for Yes or 'n'/'N': ").strip().lower()
        zoom_needed = (user_input == 'y')

        if zoom_needed:

            x_min, x_max, y_min, y_max = get_zoom_boundaries()
            plt.close(fig_temp)
        

            # Define the zoomed region
            zoom_range = (x_min, x_max)  # Adjust the range as needed
            ax_inset2 = inset_axes(ax2, width="50%", height="50%", loc='upper right')  # Inset plot
            ax_inset2.fill_between(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results["min_test_error_ELM"]]), np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results["max_test_error_ELM"]]), color='blue', alpha=0.2)
            ax_inset2.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_ELM']]), 'b-')
            ax_inset2.plot(X_axis, np.concatenate([np.array([results['test_error_incremental_ENRELM'][0]]), results['test_error_approximated_ENRELM'][0:results['test_error_ELM'].shape[0]]]), 'r-')
            ax_inset2.plot(X_axis[0:non_zeros_incremental_ENRELM_test.shape[0]], non_zeros_incremental_ENRELM_test, 'g-')
            ax_inset2.plot(X_axis[non_zeros_incremental_ENRELM_test.shape[0]:], filled_incremental_ENRELM_test, 'g--')
            ax_inset2.set_xlim(zoom_range)  # Set x-axis limits for the inset plot

            inner_plot_ylim_max = np.maximum( np.maximum(np.max(np.concatenate([np.array([1]), results["test_error_ELM"]])[int(np.floor(x_min)): int(np.ceil(x_max))]), np.max(results['test_error_approximated_ENRELM'][0:results['test_error_ELM'].shape[0]+1][int(np.floor(x_min)): int(np.ceil(x_max))])), np.max(np.concatenate((non_zeros_incremental_ENRELM_test, filled_incremental_ENRELM_test))[int(np.floor(x_min)): int(np.ceil(x_max))]) )
            inner_plot_ylim_min = np.minimum( np.minimum(np.min(np.concatenate([np.array([1]), results["test_error_ELM"]])[int(np.floor(x_min)): int(np.ceil(x_max))]), np.min(results['test_error_approximated_ENRELM'][0:results['test_error_ELM'].shape[0]+1][int(np.floor(x_min)): int(np.ceil(x_max))])), np.min(np.concatenate((non_zeros_incremental_ENRELM_test, filled_incremental_ENRELM_test))[int(np.floor(x_min)): int(np.ceil(x_max))]) ) - 0.005

            ax_inset2.set_ylim((inner_plot_ylim_min,inner_plot_ylim_max))  # Optional: match y-axis limits
            ax_inset2.grid(True)

            # Draw a rectangle on the main plot to highlight the zoomed region
            rect = Rectangle((zoom_range[0], ax_inset2.get_ylim()[0]), zoom_range[1] - zoom_range[0], ax_inset2.get_ylim()[1] - ax_inset2.get_ylim()[0],
                            edgecolor='grey', linestyle='-', linewidth=1, facecolor='none',zorder=10)
            ax2.add_patch(rect)

            # Coordinates of rectangle corners on the main plot
            rect_coords_main = [
                (zoom_range[0], ax_inset2.get_ylim()[0]),  # Bottom-left
                (zoom_range[1], ax_inset2.get_ylim()[0]),  # Bottom-right
                (zoom_range[0], ax_inset2.get_ylim()[1]),  # Top-left
                (zoom_range[1], ax_inset2.get_ylim()[1])   # Top-right
            ]

            # Coordinates of the inset corners (bottom-left and top-right corners)
            inset_coords = [
                (0, 0),                             # Bottom-left
                (1, 0),                             # Bottom-right
                (0, 1),                             # Top-left
                (1, 1)                              # Top-right
            ]

            # Add connection lines for each corner
            for main_coord, inset_coord in zip(rect_coords_main, inset_coords):
                con = ConnectionPatch(xyA=inset_coord, xyB=main_coord, coordsA="axes fraction", coordsB="data", 
                                    axesA=ax_inset2, axesB=ax2, color="grey", linestyle="--", lw=0.5)
                ax2.add_patch(con)

        else: 
            plt.close(fig_temp)

        ax.set_title(name)
        ax.set_ylabel('RMSE')
        ax.grid(True)

        ax2.set_title(name)
        ax2.set_ylabel('RMSE')
        ax2.grid(True)

    folder_path = os.path.join('results', 'images')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='lower center', ncol=2, fontsize=12)
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(os.path.join(folder_path, "image_training_real_datasets.png"))

    fig2.subplots_adjust(bottom=0.1)
    fig2.savefig(os.path.join(folder_path, "image2_test_real_datasets.png"))
    fig.show()
    fig2.show()
