#include "demo_helpers.h"

#include "tfs/checkpoint.h"
#include "tfs/random_initializer.h"
#include "tfs/tiny_language_model.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        const std::string path = "build/debug/checkpoint_demo.params";
        tfs::TinyLanguageModelConfig config;
        config.contextLength = 4;
        config.modelSize = 8;
        config.attentionSize = 8;
        config.hiddenSize = 16;

        tfs::RandomInitializer initializer(31);
        tfs::TinyLanguageModel original(config);
        original.initialize(initializer);

        const std::vector<std::size_t> inputTokens = {'d', 'e', 'r', ' '};
        const tfs::Tensor before = original.forward(inputTokens);
        tfs::saveParameters(path, original.parameters());

        tfs::TinyLanguageModel loaded(config);
        tfs::loadParameters(path, loaded.parameters());
        const tfs::Tensor after = loaded.forward(inputTokens);

        tfs::TensorValue maxDifference = 0.0f;
        for (std::size_t i = 0; i < before.size(); ++i) {
            maxDifference = std::max(maxDifference, std::fabs(before[i] - after[i]));
        }

        std::cout << "Checkpoint path: " << path << '\n';
        std::cout << "Parameter tensors: " << original.parameters().size() << '\n';
        std::cout << "Input tokens: ";
        demo::printList(inputTokens);
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Max logit difference after load: " << maxDifference << '\n';
        std::cout << std::defaultfloat;
        std::cout << "Saving and loading preserves the model output\n";
        std::cout << "The model config must still be known when loading\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
