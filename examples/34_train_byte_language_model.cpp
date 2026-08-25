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
#include <random>
#include <string>

int main(const int argc, char** argv) {
    try {
        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--train-lm") ? 2 : 1;
        if (argc != firstArg + 4 && argc != firstArg + 5) {
            std::cerr << "Usage: " << argv[0]
                      << " [--train-lm] path/to/corpus.txt maxBytes steps checkpointPath [learningRate]\n";
            std::cerr << "Set steps to 0 to train one full pass over all windows\n";
            return 1;
        }

        const std::string path = argv[firstArg];
        const std::size_t maxBytes = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 1]));
        const std::size_t requestedSteps = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 2]));
        const std::string checkpointPath = argv[firstArg + 3];
        const tfs::TensorValue learningRate =
            argc == firstArg + 5 ? demo::parseF32(argv[firstArg + 4]) : 0.03f;

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
        tfs::TensorValue lossSum = 0.0f;
        const bool fullPass = requestedSteps == 0;
        const std::size_t steps = fullPass ? dataset.size() : requestedSteps;
        std::mt19937_64 rng(123);
        std::uniform_int_distribution<std::size_t> distribution(0, dataset.size() - 1);

        const double elapsedMilliseconds = demo::measureMilliseconds([&] {
            for (std::size_t step = 0; step < steps; ++step) {
                const std::size_t index = fullPass ? step : distribution(rng);
                const tfs::TokenWindow window = dataset.windowAt(index);
                const tfs::TensorValue loss =
                    model.trainOneWindow(window.inputTokens, window.targetTokens, learningRate);

                if (step == 0) {
                    firstLoss = loss;
                }
                lastLoss = loss;
                lossSum += loss;
            }
        });

        tfs::saveParameters(checkpointPath, model.parameters());

        std::cout << "Corpus: " << path << '\n';
        std::cout << "Read bytes/tokens: " << tokens.size() << '\n';
        std::cout << "Training windows: " << dataset.size() << '\n';
        std::cout << "Requested steps: " << requestedSteps << '\n';
        std::cout << "Executed steps: " << steps << '\n';
        std::cout << "Training mode: " << (fullPass ? "one full pass over all windows" : "random sampled windows") << '\n';
        std::cout << "Token predictions trained: " << steps * config.contextLength << '\n';
        std::cout << "Context length: " << config.contextLength << '\n';
        std::cout << "Learning rate: " << learningRate << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "First loss: " << firstLoss << '\n';
        std::cout << "Last loss: " << lastLoss << '\n';
        std::cout << "Average loss: " << lossSum / static_cast<tfs::TensorValue>(steps) << '\n';
        std::cout << "Elapsed time: " << elapsedMilliseconds / 1000.0 << " s\n";
        std::cout << std::defaultfloat;
        std::cout << "Checkpoint: " << checkpointPath << '\n';
        std::cout << "This is tiny byte-level training, not a useful language model yet\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
