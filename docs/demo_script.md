# Seven-Minute Simulation Demo

Use the local app at `http://127.0.0.1:8097`. Before the audience arrives, run `python -m src.pipeline.train_resources`, then start Uvicorn and open **Live simulation**. The simulation is generated data, not a sensor connection.

| Time | Clicks and setup | What to point out |
| --- | --- | --- |
| 0:00–0:45 | Choose **Normal week** and **Start preset**. Set speed to **24h / sec**, press **Play** briefly, then pause. | The clock is simulated; the controls advance one or more hourly model ticks, not wall-clock readings. Occupancy, zone levels, and the 24-hour chart follow the selected date. |
| 0:45–1:45 | Select **Exam week**, click **Start preset**, then **Start exam surge**. | The preset lands on 10 Dec 2026, four days before the configured ETE window. Exam-window flags and the synthetic demand multiplier change together. Watch forecast utilization and peak alerts. |
| 1:45–2:40 | In **Close a zone**, select **Central Library → Quiet Study**, enter **24** hours, then click **Close a zone**. | Quiet Study becomes CLOSED with zero served occupancy. Its latent demand is redistributed to free seats in the same library first, then elsewhere; any unserved remainder is counted as overflow. |
| 2:40–3:35 | Click **Random spike**. Keep the recommender enabled and compliance at **40%**. Press **Step +1h** several times or **Play** at 1h/sec. | The event feed records scenario and redirect events. If a forecast crosses the configured near-capacity threshold, alternatives appear and some expected overflow is redirected on the following hour. Redirects are simulated student behavior, not observed compliance. |
| 3:35–4:20 | Open **Forecast** and then **Explainability** in the sidebar. | The existing historical forecast and SHAP views remain available. The simulation's recursive forecasts update hour/calendar features and predicted occupancy lags at each step; SHAP remains tied to the trained model and historical endpoint. |
| 4:20–5:10 | Return to **Live simulation**. Choose a zone from the resource selector and inspect **Resource demand · next 3 hours**. | Books, computers, study rooms, and charging have separate models and p99-derived capacity comparisons. Point out that the targets were generated from occupancy with fixed ratios and noise. |
| 5:10–6:10 | Click **Run comparison** and wait for the 14-day run to finish. | Compare total overflow and hours with any Critical zone, recommender off versus on, and the reported overflow reduction. Both runs use the same seed, exam surge, six-hour Central Library Quiet Study spike, and zone closure. Results are synthetic scenario comparisons. |
| 6:10–7:00 | Open **Video detection** and show the upload workflow, then return to the simulation or README limitations section. | Video upload is an offline demonstration, not live camera integration. A future sensor integration would need approved privacy controls, calibrated zone counts, real occupancy/resource history, and retraining before operational claims. |

## Before presenting

- Run `python -m src.sim.selftest` and confirm model artifacts load and timings are within the machine's expected range.
- Keep `data/raw/occupancy.csv` unchanged; simulation export is available from the UI as a separate download.
- Avoid describing any displayed metric as campus accuracy, actual demand, or observed recommender compliance.
- If alerts do not trigger for the selected seed/time, continue stepping during the spike or use the **Exam week** preset and retry the spike. The model and synthetic noise can move individual threshold crossings.
