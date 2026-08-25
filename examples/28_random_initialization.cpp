#include "demo_helpers.h"

#include "tfs/random_initializer.h"
#include "tfs/trainable_linear_layer.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        tfs::RandomInitializer initializer(1234);
        tfs::TrainableLinearLayer layer(3, 4);
        initializer.fillXavier(layer.getWeights().getValue(), layer.inputSize(), layer.outputSize());
        layer.getBias().getValue().fill(0.0f);

        const auto minmax = std::minmax_element(
            layer.getWeights().getValue().getValues().begin(),
            layer.getWeights().getValue().getValues().end()
        );

        std::cout << "Weight shape: ";
        demo::printList(layer.getWeights().getValue().getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Bias shape: ";
        demo::printList(layer.getBias().getValue().getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Initialized weights:\n";
        demo::printMatrix(layer.getWeights().getValue());
        std::cout << "Min/max weight: " << *minmax.first << " / " << *minmax.second << '\n';
        std::cout << "Bias values: ";
        demo::printList(layer.getBias().getValue().getValues());
        std::cout << '\n';
        std::cout << std::defaultfloat;

        std::cout << "The seed makes this initialization reproducible\n";
        std::cout << "Xavier keeps values smaller for wider layers\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
