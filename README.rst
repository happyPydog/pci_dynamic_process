Create k2 table
************************
.. code:: python


    from dypro.dynamic import NormalMeanVarChart, BrenthOptimizer
    from dypro.create_csv import create_k2

    chart = NormalMeanVarChart(alpha=0.0027)
    optimizer = BrenthOptimizer(chart, power=0.5)
    df = create_k2(
        optimizer=optimizer,
        k1_max=3,
        k1_num=0.1,
        n_max=30,
    )
    df.to_csv("xv_k2.csv", index=False, float_format="%g")