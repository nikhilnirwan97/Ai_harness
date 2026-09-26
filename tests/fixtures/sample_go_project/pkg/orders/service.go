package orders

type Order struct {
	ID     string
	Amount float64
}

type OrderService struct {
	dbName string
}

func (s *OrderService) CreateOrder(order *Order) (*Order, error) {
	return order, nil
}

func CalculateDiscount(price float64, rate float64) float64 {
	return price * (1.0 - rate)
}
