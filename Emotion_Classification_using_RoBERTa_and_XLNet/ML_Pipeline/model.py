import ktrain
import timeit
from ktrain import text
import pickle
import os
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

# create a function for training roberta and xlnet models
def create_and_train_model(train, val, transformer_model, model_name):
    model = transformer_model.get_classifier()
    model_learner_ins = None
    if model_name == "roberta":
        print("\nCompiling & Training RoBERTa for maxlen=512 & batch_size=4")
        model_learner_ins = ktrain.get_learner(model=model,
                                               train_data=train,
                                               val_data=val,                                           
                                                batch_size=4)                # Reduced the epochs, batch_size on purpose
                                                                             # Due to requirement of higher computation power                                         
        print("Model Summary: \n", model_learner_ins.model.summary())
        start_time = timeit.default_timer()
        print("\nFine Tuning RoBERTa on Human Emotion Dataset with learning rate=3e-5 and epochs=1")
        model_learner_ins.fit_onecycle(lr=3e-5, steps_per_epoch=1, epochs=1)
        stop_time = timeit.default_timer()
        print("Total time in minutes for Fine-Tuning RoBERTa on Emotion Dataset: \n", (stop_time - start_time) / 60)

    elif model_name == "xlnet":
        print("\nCompiling & Training XLNet for maxlen=128 & batch_size=4")
        model_learner_ins = ktrain.get_learner(model=model,
                                               train_data=train,
                                               val_data=val,
                                               batch_size=4)                 # Reduced the epochs, batch_size on purpose
                                                                             # Due to requirement of higher computation power 
        print("Model Summary: \n", model_learner_ins.model.summary())
        start_time = timeit.default_timer()
        print("\nFine Tuning XLNet on Human Emotion Dataset with learning rate=3e-5 and epochs=1")
        model_learner_ins.fit_onecycle(lr=3e-5, steps_per_epoch=1, epochs=1)
        stop_time = timeit.default_timer()
        print("Total time in minutes for Fine-Tuning XLNet on Emotion Dataset: \n", (stop_time - start_time) / 60)

    return model_learner_ins


def check_model_performance(model_learner_ins, class_label_names, model_name):
    print("{} Performance Metrics on Human Emotion Dataset :\n".format(model_name), model_learner_ins.validate())
    print("{} Performance Metrics on Human Emotion Dataset with Class Names :\n".format(model_name),
          model_learner_ins.validate(class_names=class_label_names))
    return None


def save_fine_tuned_model(model_learner_ins, preprocessing_var, model_name):
    save_dir = os.path.join(
        r'PATH_TO_SAVE_TRAINED_MODEL',
        f'{model_name}-emotion-predictor'
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if model_name == "roberta":
        # Save model weights and tokenizer using transformers' save_pretrained method
        model_learner_ins.model.save_pretrained(save_dir)
        
        # Save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        tokenizer.save_pretrained(save_dir)
        
        print(f"{model_name} model and tokenizer saved successfully.")
    
    elif model_name == "xlnet":
        model_learner_ins.model.save_pretrained(save_dir)
        
        tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        tokenizer.save_pretrained(save_dir)
        
        print(f"{model_name} model and tokenizer saved successfully.")


def load_model(model_name):
    load_dir = os.path.join(
        r'PATH_TO_LOAD_MODEL',
        f'{model_name}-emotion-predictor'
    )
    
    # Load model and tokenizer directly from directory
    model = TFAutoModelForSequenceClassification.from_pretrained(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    
    # Create a preprocessor based on the model name
    if model_name == "roberta":
        preproc = text.Transformer('roberta-base', maxlen=512) 
    elif model_name == "xlnet":
        preproc = text.Transformer('xlnet-base-cased', maxlen=128)  

    # Wrap the model in ktrain predictor
    predictor = ktrain.get_predictor(model, preproc=preproc)        
    print(f"{model_name.capitalize()} model loaded successfully.")
    
    return predictor


