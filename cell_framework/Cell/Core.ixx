export module Cell.Core;

export import <vector>;

export namespace Cell {
    inline int add(int a, int b) {
        return a + b;
    }

    class Engine {
    public:
        void run();
    };
}
