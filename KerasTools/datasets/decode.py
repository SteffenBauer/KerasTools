from __future__ import print_function

_mapping = {
           "mnist":
             ['Zero', 'One', 'Two', 'Three', 'Four',
              'Five', 'Six', 'Seven', 'Eight', 'Nine'],
           "fmnist": 
             ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
           "cifar10":
             ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
              'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
           "cifar100_coarse":
             [ 'Aquatic mammal', 'Fish', 
               'Flower', 'Food container', 
               'Fruit or vegetable', 'Household electrical device', 
               'Household furniture', 'Insect', 
               'Large carnivore', 'Large man-made outdoor thing', 
               'Large natural outdoor scene', 'Large omnivore or herbivore',
               'Medium-sized mammal', 'Non-insect invertebrate',
               'People', 'Reptile', 
               'Small mammal', 'Tree',
               'Vehicles Set 1', 'Vehicles Set 2'],
           "cifar100_fine":
             ['Apple', 'Aquarium fish', 'Baby', 'Bear', 'Beaver', 
              'Bed', 'Bee', 'Beetle', 'Bicycle', 'Bottle', 
              'Bowl', 'Boy', 'Bridge', 'Bus', 'Butterfly', 
              'Camel', 'Can', 'Castle', 'Caterpillar', 'Cattle', 
              'Chair', 'Chimpanzee', 'Clock', 'Cloud', 'Cockroach', 
              'Couch', 'Crab', 'Crocodile', 'Cups', 'Dinosaur', 
              'Dolphin', 'Elephant', 'Flatfish', 'Forest', 'Fox', 
              'Girl', 'Hamster', 'House', 'Kangaroo', 'Computer keyboard',
              'Lamp', 'Lawn-mower', 'Leopard', 'Lion', 'Lizard', 
              'Lobster', 'Man', 'Maple', 'Motorcycle', 'Mountain', 
              'Mouse', 'Mushrooms', 'Oak', 'Oranges', 'Orchids', 
              'Otter', 'Palm', 'Pears', 'Pickup truck', 'Pine', 
              'Plain', 'Plates', 'Poppies', 'Porcupine', 'Possum', 
              'Rabbit', 'Raccoon', 'Ray', 'Road', 'Rocket', 
              'Roses', 'Sea', 'Seal', 'Shark', 'Shrew', 
              'Skunk', 'Skyscraper', 'Snail', 'Snake', 'Spider', 
              'Squirrel', 'Streetcar', 'Sunflowers', 'Sweet peppers', 'Table', 
              'Tank', 'Telephone', 'Television', 'Tiger', 'Tractor', 
              'Train', 'Trout', 'Tulips', 'Turtle', 'Wardrobe', 
              'Whale', 'Willow', 'Wolf', 'Woman', 'Worm'],
           "reuters":
             ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
              'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
              'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
              'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
              'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead'],
           "boston":
             ["CRIM - per capita crime rate by town",
              "ZN - proportion of residential land zoned for lots over 25,000 sq.ft.",
              "INDUS - proportion of non-retail business acres per town.",
              "CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
              "NOX - nitric oxides concentration (parts per 10 million)",
              "RM - average number of rooms per dwelling",
              "AGE - proportion of owner-occupied units built prior to 1940",
              "DIS - weighted distances to five Boston employment centres",
              "RAD - index of accessibility to radial highways",
              "TAX - full-value property-tax rate per $10,000",
              "PTRATIO - pupil-teacher ratio by town",
              "B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
              "LSTAT - % lower status of the population",
              "MEDV - Median value of owner-occupied homes in $1000's"]
          }

def decode_dataset(dataset, code):
    """Returns the string description of a Keras dataset label code
   
    # Arguments
        dataset: 'mnist', 'fmnist', 'reuters', 'cifar10', 'cifar100_coarse' or 'cifar100_fine'
        code: Integer code of the label.
       
    # Returns
        The corresponding description as a string
    """
    if dataset not in _mapping.keys():
         raise ValueError('`decode_predictions` expects '
                          'a valid dataset '
                          'Requested dataset: ' + str(dataset))
    if code not in range(len(_mapping[dataset])):
         raise ValueError('Requested label code to `decode_dataset` '
                          'is out of range')
    return _mapping[dataset][code]

def decode_predictions(dataset, preds, top=3):
    """Decodes the prediction of an Keras dataset model.

    # Arguments
        dataset: 'mnist', 'fmnist', 'reuters', 'cifar10', 'cifar100_coarse' or 'cifar100_fine'
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_number, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
                    In case of invalid requested dataset.
    """
    if dataset not in _mapping.keys():
        raise ValueError('`decode_predictions` expects '
                         'a valid dataset '
                         'Requested dataset: ' + str(dataset))
    
    if len(preds.shape) != 2 or preds.shape[1] != len(_mapping[dataset]):
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, {})). '
                         'Found array with shape: {}'.format(
                           len(_mapping[dataset]), str(preds.shape)))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(i, _mapping[dataset][i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

