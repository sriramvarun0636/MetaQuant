import logging
from kiteconnect import KiteConnect

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect(api_key="03ep39d0navol1d9")

# Redirect the user to the login url obtained
# from kite.login_url(), and receive the request_token
# from the registered redirect url after the login flow.
# Once you have the request_token, obtain the access_token
# as follows.

data = kite.generate_session("CX89rqorMTQCVw4jbA6M8aPo8eN0iDbl", api_secret="2p97xj97vkfmsto9mncmy2v28ua6llks")
kite.set_access_token(data["in3bIGsGgAlCenN9WxC6ObsuqwN4KRzR"])

print("WORKING LETS FUCKING GO")
# Place an order
# try:
#     order_id = kite.place_order(tradingsymbol="INFY",
#                                 exchange=kite.EXCHANGE_NSE,
#                                 transaction_type=kite.TRANSACTION_TYPE_BUY,
#                                 quantity=1,
#                                 variety=kite.VARIETY_AMO,
#                                 order_type=kite.ORDER_TYPE_MARKET,
#                                 product=kite.PRODUCT_CNC,
#                                 validity=kite.VALIDITY_DAY)

#     logging.info("Order placed. ID is: {}".format(order_id))
# except Exception as e:
#     logging.info("Order placement failed: {}".format(e.message))

# # Fetch all orders
# kite.orders()

# # Get instruments
# kite.instruments()

# # Place an mutual fund order
# kite.place_mf_order(
#     tradingsymbol="INF090I01239",
#     transaction_type=kite.TRANSACTION_TYPE_BUY,
#     amount=5000,
#     tag="mytag"
# )

# # Cancel a mutual fund order
# kite.cancel_mf_order(order_id="order_id")

# # Get mutual fund instruments
# kite.mf_instruments()