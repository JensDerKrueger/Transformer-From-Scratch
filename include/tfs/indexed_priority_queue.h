#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tfs {

template <typename Key, typename Priority, typename PriorityCompare = std::less<Priority>>
class IndexedPriorityQueue {
public:
    bool empty() const {
        return entries.empty();
    }

    std::size_t size() const {
        return entries.size();
    }

    bool contains(const Key& key) const {
        return positions.find(key) != positions.end();
    }

    const Key& topKey() const {
        ensureNotEmpty();
        return entries.front().key;
    }

    const Priority& topPriority() const {
        ensureNotEmpty();
        return entries.front().priority;
    }

    void pushOrAssign(const Key& key, const Priority& priority) {
        const auto existing = positions.find(key);
        if (existing != positions.end()) {
            changePriorityAt(existing->second, priority);
            return;
        }

        const std::size_t index = entries.size();
        entries.push_back(Entry{key, priority});
        positions.emplace(key, index);
        swim(index);
    }

    void changePriority(const Key& key, const Priority& priority) {
        const auto existing = positions.find(key);
        if (existing == positions.end()) {
            throw std::runtime_error("Cannot change priority of missing key");
        }

        changePriorityAt(existing->second, priority);
    }

    bool erase(const Key& key) {
        const auto existing = positions.find(key);
        if (existing == positions.end()) {
            return false;
        }

        eraseAt(existing->second);
        return true;
    }

    std::pair<Key, Priority> popTop() {
        ensureNotEmpty();

        std::pair<Key, Priority> result{entries.front().key, entries.front().priority};
        eraseAt(0);
        return result;
    }

private:
    struct Entry {
        Key key;
        Priority priority;
    };

    static std::size_t parent(const std::size_t index) {
        return (index - 1) / 2;
    }

    static std::size_t leftChild(const std::size_t index) {
        return index * 2 + 1;
    }

    void ensureNotEmpty() const {
        if (entries.empty()) {
            throw std::runtime_error("IndexedPriorityQueue is empty");
        }
    }

    bool hasHigherPriority(const std::size_t lhs, const std::size_t rhs) const {
        return priorityCompare(entries[rhs].priority, entries[lhs].priority);
    }

    void swapEntries(const std::size_t lhs, const std::size_t rhs) {
        using std::swap;

        swap(entries[lhs], entries[rhs]);
        positions[entries[lhs].key] = lhs;
        positions[entries[rhs].key] = rhs;
    }

    void swim(std::size_t index) {
        while (index > 0) {
            const std::size_t parentIndex = parent(index);
            if (!hasHigherPriority(index, parentIndex)) {
                break;
            }

            swapEntries(index, parentIndex);
            index = parentIndex;
        }
    }

    void sink(std::size_t index) {
        while (true) {
            const std::size_t left = leftChild(index);
            if (left >= entries.size()) {
                break;
            }

            const std::size_t right = left + 1;
            std::size_t bestChild = left;

            if (right < entries.size() && hasHigherPriority(right, left)) {
                bestChild = right;
            }

            if (!hasHigherPriority(bestChild, index)) {
                break;
            }

            swapEntries(index, bestChild);
            index = bestChild;
        }
    }

    void repairAt(const std::size_t index) {
        if (index > 0 && hasHigherPriority(index, parent(index))) {
            swim(index);
        } else {
            sink(index);
        }
    }

    void changePriorityAt(const std::size_t index, const Priority& priority) {
        entries[index].priority = priority;
        repairAt(index);
    }

    void eraseAt(const std::size_t index) {
        const std::size_t last = entries.size() - 1;
        positions.erase(entries[index].key);

        if (index != last) {
            entries[index] = std::move(entries[last]);
            positions[entries[index].key] = index;
        }

        entries.pop_back();

        if (index < entries.size()) {
            repairAt(index);
        }
    }

    std::vector<Entry> entries;
    std::unordered_map<Key, std::size_t> positions;
    PriorityCompare priorityCompare;
};

} // namespace tfs
