#include "demo_helpers.h"

#include "tfs/checkpoint.h"
#include "tfs/random_initializer.h"
#include "tfs/tiny_language_model.h"
#include "tfs/token_dataset.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(const int argc, char** argv) {
    try {
        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--train-lm") ? 2 : 1;
        if (argc != firstArg + 4) {
            std::cerr << "Usage: " << argv[0]
                      << " [--train-lm] path/to/corpus.txt maxBytes steps checkpointPath\n";
            return 1;
        }

        const std::string path = argv[firstArg];
        const std::size_t maxBytes = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 1]));
        const std::size_t steps = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 2]));
        const std::string checkpointPath = argv[firstArg + 3];

        tfs::TinyLanguageModelConfig config;
        config.contextLength = 16;
        config.modelSize = 16;
        config.attentionSize = 16;
        config.hiddenSize = 32;

        const std::vector<std::size_t> tokens = tfs::readByteTokensFromFile(path, maxBytes);
        const tfs::TokenWindowDataset dataset(tokens, config.contextLength);

        tfs::RandomInitializer initializer(23);
        tfs::TinyLanguageModel model(config);
        model.initialize(initializer);

        tfs::TensorValue firstLoss = 0.0f;
        tfs::TensorValue lastLoss = 0.0f;
        const tfs::TensorValue learningRate = 0.03f;

        for (std::size_t step = 0; step < steps; ++step) {
            const std::size_t index = (step * 37) % dataset.size();
            const tfs::TokenWindow window = dataset.windowAt(index);
            const tfs::TensorValue loss = model.trainOneWindow(window.inputTokens, window.targetTokens, learningRate);

            if (step == 0) {
                firstLoss = loss;
            }
            lastLoss = loss;
        }

        tfs::saveParameters(checkpointPath, model.parameters());

        std::cout << "Corpus: " << path << '\n';
        std::cout << "Read bytes/tokens: " << tokens.size() << '\n';
        std::cout << "Training windows: " << dataset.size() << '\n';
        std::cout << "Steps: " << steps << '\n';
        std::cout << "Context length: " << config.contextLength << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "First loss: " << firstLoss << '\n';
        std::cout << "Last loss: " << lastLoss << '\n';
        std::cout << std::defaultfloat;
        std::cout << "Checkpoint: " << checkpointPath << '\n';
        std::cout << "This is tiny byte-level training, not a useful language model yet\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
