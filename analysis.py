
def analysis():
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    all_predicted = [
        1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 1, 0, 0, 2, 1, 1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    all_actual = [
        1, 1, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 2, 1, 2, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    confusion_matrix(all_actual, all_predicted)
    f1_weighted = precision_recall_fscore_support(all_actual, all_predicted, average='weighted')
    print(f1_weighted)

    f1_weighted = precision_recall_fscore_support(all_actual, all_predicted, average='micro')
    print(f1_weighted)

    f1_weighted = precision_recall_fscore_support(all_actual, all_predicted, average='macro')
    print(f1_weighted)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=2345)
    # hello_world()
    analysis()