import matplotlib.pyplot
from CrossValidation import *
from DataSet import *
from utils import *


def testDataset(dataset):
    print("Testing dataset:"+dataset.name)
    perc = [0, 0.1, 0.2, 0.5]
    results = []
    for x in perc:
        if x == 0:
            print("running test:no value removed")
        else:
            print("running test:value removed with probability " + str(x))
        newDataset = removeRadomValues(dataset, x)
        replaceMissingValue(newDataset)
        err_test = cross_validation(newDataset)
        err_test = round(err_test * 100, 4)
        print("Testing error:"+str(err_test)+"%")
        accuracy = round(100 - err_test, 4)
        results.append(accuracy)
    print()
    return results


if __name__ == '__main__':
    # create data sets

    car = parse_file("car.txt")
    carDataset = DataSet(name="Car Evaluation", examples=car, target=6)

    tictactoe = parse_file("tic-tac-toe.txt")
    tictactoeDataset = DataSet(name="Tic-Tac-Toe", examples=tictactoe, target=9)

    bank = parse_file("QualitativeBankruptcy.txt")
    bankDataset = DataSet(name="Qualitative Bankruptcy", examples=bank, target=6)

    # test data sets

    carResult = testDataset(carDataset)
    tttResult = testDataset(tictactoeDataset)
    bankResult = testDataset(bankDataset)

    # create ad show graphic
    xValues = [0.0, 0.1, 0.2, 0.5]
    matplotlib.pyplot.plot(xValues, carResult, label=carDataset.name)
    matplotlib.pyplot.plot(xValues, tttResult, label=tictactoeDataset.name)
    matplotlib.pyplot.plot(xValues, bankResult, label=bankDataset.name)
    matplotlib.pyplot.title("Alberi di decisione con dati mancanti")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
