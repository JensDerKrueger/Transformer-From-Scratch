#include "demo_helpers.h"

#include "tfs/cross_entropy_loss.h"
#include "tfs/sgd_optimizer.h"
#include "tfs/trainable_linear_layer.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        tfs::TrainableLinearLayer classifier(
            2,
            3,
            {0.10f, -0.20f, 0.30f,
             0.40f, 0.00f, -0.10f},
            {0.00f, 0.00f, 0.00f}
        );
        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({3, 2}),
            {1.00f, 0.00f,
             0.00f, 1.00f,
             1.00f, 1.00f}
        );
        const std::vector<std::size_t> targets = {0, 1, 2};
        const tfs::TensorValue learningRate = 0.50f;

        tfs::zeroGradients(classifier.parameters());
        const tfs::Tensor logitsBefore = classifier.forward(input);
        const tfs::CrossEntropyResult lossBefore = tfs::crossEntropyLoss(logitsBefore, targets);
        classifier.backward(input, lossBefore.gradient);
        tfs::sgdStep(classifier.parameters(), learningRate);
        const tfs::Tensor logitsAfter = classifier.forward(input);
        const tfs::CrossEntropyResult lossAfter = tfs::crossEntropyLoss(logitsAfter, targets);

        std::cout << "Targets: ";
        demo::printList(targets);
        std::cout << '\n';
        std::cout << "Learning rate: " << learningRate << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Input:\n";
        demo::printMatrix(input);
        std::cout << "Logits before SGD:\n";
        demo::printMatrix(logitsBefore);
        std::cout << "Loss before SGD: " << lossBefore.loss << '\n';
        std::cout << "Weight gradient:\n";
        demo::printMatrix(classifier.getWeights().getGradient());
        std::cout << "Bias gradient: ";
        demo::printList(classifier.getBias().getGradient().getValues());
        std::cout << '\n';
        std::cout << "Weights after SGD:\n";
        demo::printMatrix(classifier.getWeights().getValue());
        std::cout << "Logits after SGD:\n";
        demo::printMatrix(logitsAfter);
        std::cout << "Loss after SGD: " << lossAfter.loss << '\n';
        std::cout << std::defaultfloat;

        std::cout << "One SGD step moves parameters opposite to the gradient\n";
        std::cout << "The loss should usually go down for a small enough learning rate\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
