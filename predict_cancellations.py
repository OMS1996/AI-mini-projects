from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

def predict_cancellations(user_interaction_df):
    # Combine input columns into a single features vector column
    assembler = VectorAssembler(
        inputCols=["month_interaction_count", "week_interaction_count", "day_interaction_count"], 
        outputCol="features"
    )
    
    # Transform the DataFrame to include the features vector
    features_df = assembler.transform(user_interaction_df)
    
    # Add the label column to the DataFrame
    features_df = features_df.withColumn("label", features_df["cancelled_within_week"])
    
    # Initialize logistic regression model
    lr_model = LogisticRegression(maxIter=10, threshold=0.6, elasticNetParam=1, regParam=0.1)
    
    # Train the model
    trained_lr_model = lr_model.fit(features_df)
    
    # Generate predictions
    predictions_df = trained_lr_model.transform(features_df)
    
    # Select the relevant columns for output
    predictions_df = predictions_df.select(["user_id", "rawPrediction", "probability", "prediction"])
    
    # Return the predictions DataFrame
    return predictions_df
