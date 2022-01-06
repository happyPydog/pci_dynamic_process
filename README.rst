dypro: Dynamic Process with Mean Shift and Variance Change
====================================================================


Dynamic model
************************
Let us consider $\bar_{X}, R$ control chart and $\alpha = 0.0027$ as an example.
.. code:: python



    from dypro.dynamic import NormalMeanRChart

    k1 = 1.2
    k2 = 1.4
    n = 5
    chart = NormalMeanRChart(alpha=0.0027)
    print(f"Detection Power = {chart.power(k1=k1, k2=k2, n=n):.4f}")

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

