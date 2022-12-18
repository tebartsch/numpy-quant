import plotext as plt


def profile_results_plot(profile_results: dict[str, float],
                         q_profile_results: dict[str, float]):
    show = 5  # Number of entries to show

    q_profile_results = {k: v for k, v in sorted(q_profile_results.items(), key=lambda item: -item[1])}
    adj_profile_results = {}
    for key in q_profile_results:
        if key in profile_results:
            adj_profile_results[key] = profile_results[key]
        else:
            adj_profile_results[key] = 0.0

    plt.axes_color('default')
    plt.canvas_color('default')
    plt.ticks_color('default')
    labels = list(q_profile_results)
    height = list(adj_profile_results.values())
    qheight = list(q_profile_results.values())
    plt.simple_multiple_bar(labels[:show], [qheight[:show], height[:show]], labels=["quantized", "float32"])
    plt.title("Profile quantized vs. float")
    plt.show()
    plt.clear_figure()
