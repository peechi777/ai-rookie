import unittest

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

class TestTrie(unittest.TestCase):
    def test_trie_functions(self):
        trie = Trie()
        trie.insert("apple")
        trie.insert("app")
        trie.insert("banana")
        trie.insert("band")
        
        self.assertTrue(trie.search("apple"))
        self.assertTrue(trie.search("app"))
        self.assertFalse(trie.search("appl"))
        
        
        self.assertTrue(trie.starts_with("app"))
        self.assertTrue(trie.starts_with("ban"))
        self.assertTrue(trie.starts_with("band"))
        self.assertFalse(trie.starts_with("bat"))

if __name__ == "__main__":
    unittest.main()