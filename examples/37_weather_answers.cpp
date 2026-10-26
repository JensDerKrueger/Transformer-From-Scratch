#include "demo_helpers.h"

#include "tfs/checkpoint.h"
#include "tfs/response_training.h"
#include "tfs/word_tokenizer.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct WeatherExample {
    std::string prompt;
    std::string answer;
    bool heldOut;
};

std::vector<WeatherExample> makeExamples() {
    const std::vector<std::string> cities{"Berlin", "Hamburg", "Bonn", "Bremen", "Kiel", "Ulm"};
    const std::vector<std::string> conditions{"Regen", "Sonne", "Schnee", "Wolken"};
    const std::vector<std::string> answers{
        "Nimm einen Schirm mit.", "Nimm eine Sonnenbrille mit.",
        "Nimm eine Winterjacke mit.", "Nimm eine leichte Jacke mit."
    };
    const std::vector<std::string> questions{"Was brauche ich?", "Was soll ich mitnehmen?"};
    std::vector<WeatherExample> examples;
    for (std::size_t city = 0; city < cities.size(); ++city) {
        for (std::size_t condition = 0; condition < conditions.size(); ++condition) {
            for (const std::string& question : questions) {
                const std::string prompt = "In " + cities[city] + " gibt es " + conditions[condition]
                    + ". " + question + " Antwort:";
                // Both question variants for this city/weather pair stay out of training.
                examples.push_back({prompt, answers[condition], condition == city % conditions.size()});
            }
        }
    }
    return examples;
}

tfs::TinyLanguageModelConfig makeConfig(const tfs::WordTokenizer& tokenizer) {
    tfs::TinyLanguageModelConfig config;
    config.vocabSize = tokenizer.size();
    config.contextLength = 32;
    config.modelSize = 24;
    config.attentionSize = 24;
    config.hiddenSize = 48;
    return config;
}

std::vector<std::size_t> answerTokens(const tfs::WordTokenizer& tokenizer, const std::string& answer) {
    std::vector<std::size_t> tokens = tokenizer.encode(answer);
    tokens.push_back(tokenizer.endToken());
    return tokens;
}

std::size_t evaluate(
    const tfs::TinyLanguageModel& model,
    const tfs::WordTokenizer& tokenizer,
    const std::vector<WeatherExample>& examples,
    const bool heldOut,
    const bool printAnswers
) {
    std::size_t correct = 0;
    std::size_t count = 0;
    for (const WeatherExample& example : examples) {
        if (example.heldOut != heldOut) { continue; }
        const std::vector<std::size_t> generated = tfs::generateResponse(
            model, tokenizer.encode(example.prompt), tokenizer.endToken(), 12);
        // Compare tokens including EOS: even an unterminated correct-looking sentence fails.
        const bool matches = generated == answerTokens(tokenizer, example.answer);
        correct += matches ? 1 : 0;
        ++count;
        if (printAnswers) {
            std::cout << example.prompt << '\n' << "  " << tokenizer.decode(generated)
                      << (matches ? " [OK]" : " [FAIL]") << '\n';
        }
    }
    std::cout << (heldOut ? "Held-out" : "Training") << " exact answers: " << correct << '/' << count << '\n';
    return correct;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " train checkpoint [epochs]\n"
                      << "       " << argv[0] << " ask checkpoint \"prompt\"\n"
                      << "       " << argv[0] << " interactive checkpoint\n";
            return 1;
        }
        const std::string mode = argv[1];
        const std::string checkpointPath = argv[2];
        if ((mode == "train" && argc != 3 && argc != 4)
            || (mode == "ask" && argc != 4) || (mode == "interactive" && argc != 3)
            || (mode != "train" && mode != "ask" && mode != "interactive")) {
            throw std::runtime_error("Invalid weather demo arguments");
        }
        const std::vector<WeatherExample> examples = makeExamples();
        std::vector<std::string> texts;
        std::vector<std::size_t> trainingIndices;
        for (std::size_t index = 0; index < examples.size(); ++index) {
            if (!examples[index].heldOut) {
                texts.push_back(examples[index].prompt);
                texts.push_back(examples[index].answer);
                trainingIndices.push_back(index);
            }
        }
        const tfs::WordTokenizer tokenizer(texts);
        tfs::TinyLanguageModel model(makeConfig(tokenizer));

        if (mode == "train") {
            const std::size_t epochs = argc == 4 ? static_cast<std::size_t>(demo::parseU64(argv[3])) : 200;
            if (epochs == 0) { throw std::runtime_error("Epoch count must be positive"); }
            tfs::RandomInitializer initializer(23);
            model.initialize(initializer);
            std::mt19937 generator(123);
            std::size_t parameterCount = 0;
            for (const tfs::Parameter* const parameter : model.parameters()) {
                parameterCount += parameter->getValue().size();
            }
            std::cout << "Vocabulary: " << tokenizer.size() << " word/punctuation/EOS tokens\n"
                      << "Parameters: " << parameterCount << '\n'
                      << "Training examples: " << trainingIndices.size() << "; held-out examples: "
                      << examples.size() - trainingIndices.size() << '\n';
            evaluate(model, tokenizer, examples, true, false);
            const double elapsed = demo::measureMilliseconds([&] {
                for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                    std::shuffle(trainingIndices.begin(), trainingIndices.end(), generator);
                    double loss = 0.0;
                    for (const std::size_t index : trainingIndices) {
                        const WeatherExample& example = examples[index];
                        loss += tfs::trainResponse(model, tokenizer.encode(example.prompt),
                            answerTokens(tokenizer, example.answer), 0.03f);
                    }
                    if (epoch == 0 || (epoch + 1) % 25 == 0 || epoch + 1 == epochs) {
                        std::cout << "Epoch " << epoch + 1 << '/' << epochs << ": response loss "
                                  << std::fixed << std::setprecision(4) << loss / trainingIndices.size()
                                  << std::endl;
                    }
                }
            });
            evaluate(model, tokenizer, examples, false, false);
            const std::size_t correct = evaluate(model, tokenizer, examples, true, true);
            std::cout << "Training time: " << std::fixed << std::setprecision(3) << elapsed / 1000.0 << " s\n";
            tfs::saveParameters(checkpointPath, model.parameters());
            std::cout << "Checkpoint: " << checkpointPath << '\n';
            return correct == examples.size() - trainingIndices.size() ? 0 : 2;
        }

        tfs::loadParameters(checkpointPath, model.parameters());
        const auto answer = [&](const std::string& prompt) {
            std::cout << tokenizer.decode(tfs::generateResponse(
                model, tokenizer.encode(prompt), tokenizer.endToken(), 12)) << '\n';
        };
        if (mode == "ask") {
            answer(argv[3]);
        } else {
            std::cout << "Example: In Berlin gibt es Regen. Was brauche ich? Antwort:\n"
                      << "Closed vocabulary; use the demonstrated sentence forms. Empty line exits.\n";
            std::string prompt;
            while (std::cout << "> " && std::getline(std::cin, prompt) && !prompt.empty()) {
                try { answer(prompt); }
                catch (const std::exception& error) { std::cout << error.what() << '\n'; }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
