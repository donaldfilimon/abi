import SwiftUI

struct ContentView: View {
    @State private var selectedSection: String = "Quiz"
    @State private var questionIndex: Int = 0
    @State private var score: Int = 0
    @State private var showAnswer: Bool = false
    @State private var items: [String] = ["Item 1", "Item 2", "Item 3"]
    @State private var timestamps: [Date] = [Date(), Date().addingTimeInterval(-3600), Date().addingTimeInterval(-7200)]
    
    private let questions = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("What color is the sky?", "Blue")
    ]
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Control Panel")
                    .font(.largeTitle)
                    .bold()
                    .padding(.top)
                
                Text("Status: All systems operational")
                    .font(.headline)
                    .padding(.bottom)
                
                Picker("Select Section", selection: $selectedSection) {
                    Text("Quiz").tag("Quiz")
                    Text("Items").tag("Items")
                    Text("Status").tag("Status")
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.bottom)
                
                Divider()
                
                Group {
                    if selectedSection == "Quiz" {
                        quizView
                    } else if selectedSection == "Items" {
                        itemsView
                    } else if selectedSection == "Status" {
                        statusView
                    }
                }
                .animation(.default, value: selectedSection)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Control Panel")
        }
    }
    
    var quizView: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Quiz")
                .font(.title2)
                .bold()
            
            Text("Score: \(score)")
                .font(.headline)
            
            Text(questions[questionIndex].0)
                .font(.title3)
            
            if showAnswer {
                Text("Answer: \(questions[questionIndex].1)")
                    .foregroundColor(.green)
                    .font(.headline)
            }
            
            HStack {
                Button("Show Answer") {
                    showAnswer = true
                }
                .disabled(showAnswer)
                
                Spacer()
                
                Button("Correct") {
                    score += 1
                    nextQuestion()
                }
                .disabled(!showAnswer)
                
                Button("Incorrect") {
                    nextQuestion()
                }
                .disabled(!showAnswer)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(10)
    }
    
    var itemsView: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Items")
                .font(.title2)
                .bold()
            
            List {
                ForEach(items.indices, id: \.self) { index in
                    VStack(alignment: .leading) {
                        Text(items[index])
                            .font(.headline)
                        Text("Timestamp: \(timestamps[index], formatter: dateFormatter)")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    .padding(.vertical, 5)
                }
            }
            .listStyle(PlainListStyle())
        }
    }
    
    var statusView: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("System Status")
                .font(.title2)
                .bold()
            
            Text("All systems operational")
                .font(.body)
                .foregroundColor(.green)
            
            Text("No alerts or warnings.")
                .font(.body)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(10)
    }
    
    func nextQuestion() {
        showAnswer = false
        questionIndex = (questionIndex + 1) % questions.count
    }
    
    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }
}
