#include "tfs/response_training.h"
#include "tfs/word_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        const tfs::WordTokenizer tokenizer({"In Berlin gibt es Regen. Antwort: Nimm einen Schirm mit."});
        const std::string text = "In Berlin gibt es Regen. Antwort: Nimm einen Schirm mit.";
        if (tokenizer.decode(tokenizer.encode(text)) != text) {
            throw std::runtime_error("Word tokenizer round trip failed");
        }
        bool rejectedUnknown = false;
        try { tokenizer.encode("unknown"); }
        catch (const std::runtime_error&) { rejectedUnknown = true; }
        if (!rejectedUnknown) { throw std::runtime_error("Unknown word was not rejected"); }

        const tfs::TinyLanguageModelConfig config{8, 8, 4, 4, 8};
        tfs::TinyLanguageModel model(config);
        tfs::RandomInitializer initializer(42);
        model.initialize(initializer);
        const std::vector<std::size_t> input{1, 2, 3};
        const std::vector<std::size_t> response{3, 0};
        const tfs::TinyLanguageModelForwardResult forward = model.forwardDetailed(input);
        const tfs::ResponseLossResult loss = tfs::responseLoss(forward.logits, 2, response);
        for (std::size_t column = 0; column < config.vocabSize; ++column) {
            if (loss.gradient.data()[column] != 0.0f) {
                throw std::runtime_error("Prompt-only row has a loss gradient");
            }
        }
        const std::vector<tfs::Parameter*> parameters = model.parameters();
        tfs::zeroGradients(parameters);
        model.backward(forward, loss.gradient);
        std::size_t checked = 0;
        tfs::TensorValue maxError = 0.0f;
        for (tfs::Parameter* const parameter : parameters) {
            tfs::Tensor& value = parameter->getValue();
            const std::vector<std::size_t> indices{0, std::min<std::size_t>(7, value.size() - 1),
                value.size() / 2, value.size() - 1};
            for (const std::size_t index : indices) {
                const tfs::TensorValue original = value[index];
                const tfs::TensorValue epsilon = 0.001f;
                value[index] = original + epsilon;
                const tfs::TensorValue plus = tfs::responseLoss(model.forward(input), 2, response).loss;
                value[index] = original - epsilon;
                const tfs::TensorValue minus = tfs::responseLoss(model.forward(input), 2, response).loss;
                value[index] = original;
                const tfs::TensorValue numerical = (plus - minus) / (2.0f * epsilon);
                const tfs::TensorValue analytical = parameter->getGradient()[index];
                const tfs::TensorValue error = std::abs(numerical - analytical);
                maxError = std::max(maxError, error);
                if (!std::isfinite(error) || error > 0.005f + 0.02f * std::abs(numerical)) {
                    throw std::runtime_error("Response loss finite-difference gradient check failed");
                }
                ++checked;
            }
        }
        std::cout << "Tokenizer round trip and unknown-word rejection: OK\n"
                  << "Prompt loss mask: OK\n"
                  << "End-to-end finite-difference gradients: " << checked
                  << " checked, max absolute error " << maxError << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
