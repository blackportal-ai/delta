use crate::algorithms::{LinearRegression, TrainingMetrics};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ndarray::{Array1, Array2};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    symbols::Marker,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph, Tabs},
};
use std::io::{self, Stdout};
use std::time::Duration;

pub struct Controls;

impl Controls {
    fn render(f: &mut Frame, area: Rect) {
        let items = vec![
            ListItem::new("q: Quit"),
            ListItem::new("t: Toggle Plot Tab"),
            ListItem::new("p: Pause/Resume"),
            ListItem::new("s: Start/Stop"),
        ];
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::White));
        f.render_widget(list, area);
    }
}

pub struct Metrics {
    loss: f64,
}

impl Metrics {
    fn new() -> Self {
        Metrics { loss: 0.0 }
    }

    fn update(&mut self, loss: f64) {
        self.loss = loss;
    }

    fn render(&self, f: &mut Frame, area: Rect) {
        let cpu_usage = 8.9; // Placeholder
        let text = format!("CPU Usage: {:.1}%\nCPU Temp: N/A\nLoss: {:.6}", cpu_usage, self.loss);
        let paragraph = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .style(Style::default().fg(Color::White));
        f.render_widget(paragraph, area);
    }
}

pub struct Plots {
    loss_data: Vec<(f64, f64)>, // (epoch, loss)
    selected_tab: usize,
    max_points: usize,
}

impl Plots {
    fn new() -> Self {
        Plots { loss_data: Vec::new(), selected_tab: 0, max_points: 100 }
    }

    fn update(&mut self, metrics: &[TrainingMetrics]) {
        if let Some(latest) = metrics.last() {
            let loss = latest.loss.clamp(0.0, 10.0); // Cap MSE loss
            self.loss_data.push((self.loss_data.len() as f64, loss));

            if self.loss_data.len() > self.max_points {
                self.loss_data.drain(0..1); // Remove oldest
            }

            // Scale x-coordinates to [0, max_points-1]
            let len = self.loss_data.len() as f64;
            let max_x = (self.max_points - 1) as f64;
            self.loss_data.iter_mut().enumerate().for_each(|(i, (x, _))| {
                *x = if len <= 1.0 { 0.0 } else { (i as f64 / (len - 1.0)) * max_x };
            });
        }
    }

    fn toggle_tab(&mut self) {
        self.selected_tab = 0; // Single tab
    }

    fn render(&self, f: &mut Frame, area: Rect) {
        let titles = vec!["Loss"];
        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL).title("Plots"))
            .select(self.selected_tab)
            .style(Style::default().fg(Color::White));
        f.render_widget(tabs, area);

        let inner_area =
            Rect { x: area.x + 1, y: area.y + 2, width: area.width - 2, height: area.height - 3 };

        let x_bounds = [0.0, (self.max_points - 1) as f64];
        let y_bounds = [0.0, 10.0]; // Cap MSE loss

        let dataset = Dataset::default()
            .name("Loss Curve")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&self.loss_data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title("Loss Curve"))
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::White))
                    .bounds(x_bounds),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::White))
                    .bounds(y_bounds),
            );
        f.render_widget(chart, inner_area);
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AppState {
    Idle,
    Training,
    Paused,
    Testing,
}

pub struct App {
    pub model: LinearRegression,
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    x_test: Array2<f64>,
    y_test: Array1<f64>,
    learning_rate: f64,
    epochs: usize,
    metrics: Metrics,
    plots: Plots,
    state: AppState,
    current_epoch: usize,
}

impl App {
    fn new(
        model: LinearRegression,
        x_train: Array2<f64>,
        y_train: Array1<f64>,
        x_test: Array2<f64>,
        y_test: Array1<f64>,
        learning_rate: f64,
        epochs: usize,
    ) -> Self {
        App {
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            learning_rate,
            epochs,
            metrics: Metrics::new(),
            plots: Plots::new(),
            state: AppState::Idle,
            current_epoch: 0,
        }
    }

    fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('t') => self.plots.toggle_tab(),
                        KeyCode::Char('p') => {
                            self.state = match self.state {
                                AppState::Training => AppState::Paused,
                                AppState::Paused => AppState::Training,
                                other => other,
                            };
                        }
                        KeyCode::Char('s') => {
                            self.state = match self.state {
                                AppState::Idle => {
                                    self.current_epoch = 0;
                                    AppState::Training
                                }
                                AppState::Training | AppState::Paused => AppState::Idle,
                                AppState::Testing => AppState::Idle,
                            };
                        }
                        _ => {}
                    }
                }
            }

            if self.state == AppState::Training {
                self.train_one_epoch();
            }
        }
    }

    fn train_one_epoch(&mut self) {
        if self.current_epoch < self.epochs && self.state == AppState::Training {
            self.model
                .fit(&self.x_train, &self.y_train, self.learning_rate, 1)
                .expect("Fit failed");
            self.current_epoch += 1;

            let metrics = self.model.metrics();
            if let Some(last) = metrics.last() {
                self.metrics.update(last.loss);
                self.plots.update(metrics);
            }
        } else if self.current_epoch >= self.epochs && self.state == AppState::Training {
            self.state = AppState::Testing;
            let predictions = self.model.predict(&self.x_test).expect("Predict failed");
            let test_loss = self
                .model
                .calculate_loss(&predictions, &self.y_test)
                .expect("Loss calculation failed");
            self.metrics.update(test_loss);
            self.state = AppState::Idle;
        }
    }

    fn render(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(f.area());

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Controls
                Constraint::Length(8), // Metrics
                Constraint::Min(0),    // Status
            ])
            .split(chunks[0]);

        Controls::render(f, left_chunks[0]);
        self.metrics.render(f, left_chunks[1]);

        let status_text = match self.state {
            AppState::Idle => "Idle".to_string(),
            AppState::Training => format!("Training: Epoch {}/{}", self.current_epoch, self.epochs),
            AppState::Paused => format!("Paused: Epoch {}/{}", self.current_epoch, self.epochs),
            AppState::Testing => "Testing".to_string(),
        };
        let status =
            Paragraph::new(format!("Status: {}\nItems: {}", status_text, self.x_train.nrows()))
                .block(Block::default().borders(Borders::ALL).title("Status"))
                .style(Style::default().fg(Color::White));
        f.render_widget(status, left_chunks[2]);

        self.plots.render(f, chunks[1]);
    }
}

pub fn init_tui(
    model: LinearRegression,
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    x_test: Array2<f64>,
    y_test: Array1<f64>,
    learning_rate: f64,
    epochs: usize,
) -> io::Result<LinearRegression> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(model, x_train, y_train, x_test, y_test, learning_rate, epochs);
    app.run(&mut terminal)?;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(app.model)
}
