#include "demo_helpers.h"

#include "tfs/cross_entropy_loss.h"
#include "tfs/random_initializer.h"
#include "tfs/tiny_language_model.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        tfs::TinyLanguageModelConfig config;
        config.contextLength = 4;
        config.modelSize = 8;
        config.attentionSize = 8;
        config.hiddenSize = 16;

        tfs::RandomInitializer initializer(17);
        tfs::TinyLanguageModel model(config);
        model.initialize(initializer);

        const std::vector<std::size_t> inputTokens = {'d', 'e', 'r', ' '};
        const std::vector<std::size_t> targetTokens = {'e', 'r', ' ', 'f'};
        const tfs::TinyLanguageModelForwardResult forward = model.forwardDetailed(inputTokens);
        const tfs::CrossEntropyResult loss = tfs::crossEntropyLoss(forward.logits, targetTokens);
        model.backward(forward, loss.gradient);

        std::cout << "Input tokens: ";
        demo::printList(inputTokens);
        std::cout << '\n';
        std::cout << "Target tokens: ";
        demo::printList(targetTokens);
        std::cout << '\n';
        std::cout << "Logit shape: ";
        demo::printList(forward.logits.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "First logit row, first 8 entries:";
        for (std::size_t i = 0; i < 8; ++i) {
            std::cout << ' ' << forward.logits[i];
        }
        std::cout << '\n';
        std::cout << "Loss: " << loss.loss << '\n';
        std::cout << std::defaultfloat;

        std::cout << "The model produces one vocabulary-sized logit row per input token\n";
        std::cout << "A backward pass now reaches embeddings, attention, feed-forward and head parameters\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
