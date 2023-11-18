
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.model.d_linear.estimator import DLinearEstimator

from gluonts.evaluation import make_evaluation_predictions, Evaluator
import matplotlib.pyplot as plt

prediction_length = 24
context_length = prediction_length*2
batch_size = 128
num_batches_per_epoch = 100
epochs = 50
scaling = "std"


encoder_layers=2
decoder_layers=2
d_model=16



dataset = get_dataset("tourism_quarterly", prediction_length=5)
freq = dataset.metadata.freq
prediction_length = dataset.metadata.prediction_length

train_dataset = dataset.train
test_dataset = dataset.test



# Define the DLinear model with the same parameters as the Autoformer model
estimator = DLinearEstimator(
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length*2,
    scaling=scaling,
    hidden_dimension=2,
    
    batch_size=batch_size,
    num_batches_per_epoch=num_batches_per_epoch,
    trainer_kwargs=dict(max_epochs=epochs)
)
predictor = estimator.train(
    
    training_data=train_dataset, 
    cache_data=True, 
    shuffle_buffer_length=1024,
    epochs = 100
)



forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
)

d_linear_forecasts = list(forecast_it)
d_linear_tss = list(ts_it)

evaluator = Evaluator()

agg_metrics, _ = evaluator(iter(d_linear_tss), iter(d_linear_forecasts))

dlinear_mase = agg_metrics["MASE"]
print(f"DLinear MASE: {dlinear_mase:.3f}")

dlinear_mape = agg_metrics["MAPE"]
print(f"DLinear MAPE: {dlinear_mape:.3f}")

def plot_gluonts(index):
    plt.plot(d_linear_tss[index][-4 * dataset.metadata.prediction_length:].to_timestamp(), label="target")
    d_linear_forecasts[index].plot(show_label=True,  color='g')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()
    
plot_gluonts(4)
    
